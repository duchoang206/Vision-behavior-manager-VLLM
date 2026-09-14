import 'server-only';

import { createHmac, randomBytes, scrypt, timingSafeEqual } from 'node:crypto';
import { promisify } from 'node:util';
import { cookies } from 'next/headers';

export const SESSION_COOKIE = 'rskyview_session';
const PASSWORD_SALT = process.env.RSKYVIEW_PASSWORD_SALT || 'r-skyview-admin-v1';
const DEFAULT_PASSWORD_HASH = '5b46b622d7bef1357804f3040dfdb64235e4152a99b4660add351425e9d2388f73eace07357868ee6fb1f9c6714af8dd33e52560b34740577da3f5fe011f3bcf';
const SESSION_DURATION = 8 * 60 * 60;
const deriveKey = promisify(scrypt);

export type SessionPayload = {
  username: string;
  issuedAt: number;
  expiresAt: number;
  nonce: string;
};

function signingKey() {
  const secret = process.env.RSKYVIEW_SESSION_SECRET;
  if (!secret || secret.length < 32) throw new Error('RSKYVIEW_SESSION_SECRET must contain at least 32 characters.');
  return createHmac('sha256', secret)
    .update(`${configuredUsername()}:${PASSWORD_SALT}:${process.env.RSKYVIEW_ADMIN_PASSWORD_HASH || DEFAULT_PASSWORD_HASH}`)
    .digest();
}

export function configuredUsername() {
  return process.env.RSKYVIEW_ADMIN_USERNAME || 'admin';
}

export async function credentialsMatch(username: string, password: string) {
  const expectedUsername = configuredUsername();
  const expectedHash = process.env.RSKYVIEW_ADMIN_PASSWORD_HASH || DEFAULT_PASSWORD_HASH;
  const derivedKey = (await deriveKey(password, PASSWORD_SALT, 64)) as Buffer;
  const expectedKey = Buffer.from(expectedHash, 'hex');
  return username === expectedUsername && expectedKey.length === derivedKey.length && timingSafeEqual(expectedKey, derivedKey);
}

export function createSessionToken(username: string) {
  const issuedAt = Math.floor(Date.now() / 1000);
  const expiresAt = issuedAt + SESSION_DURATION;
  const payload: SessionPayload = { username, issuedAt, expiresAt, nonce: randomBytes(16).toString('hex') };
  const encodedPayload = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const signature = createHmac('sha256', signingKey()).update(encodedPayload).digest('base64url');
  return { token: `${encodedPayload}.${signature}`, maxAge: expiresAt - issuedAt };
}

export function verifySessionToken(token: string | undefined): SessionPayload | null {
  if (!token || token.length > 1024) return null;
  const parts = token.split('.');
  if (parts.length !== 2 || !parts.every(part => /^[A-Za-z0-9_-]+$/.test(part))) return null;
  const [encodedPayload, encodedSignature] = parts;
  try {
    const expectedSignature = createHmac('sha256', signingKey()).update(encodedPayload).digest();
    const receivedSignature = Buffer.from(encodedSignature, 'base64url');
    if (expectedSignature.length !== receivedSignature.length || !timingSafeEqual(expectedSignature, receivedSignature)) return null;
    const payload = JSON.parse(Buffer.from(encodedPayload, 'base64url').toString('utf8')) as SessionPayload;
    const now = Math.floor(Date.now() / 1000);
    if (!payload || payload.username !== configuredUsername() || !Number.isSafeInteger(payload.issuedAt) ||
      !Number.isSafeInteger(payload.expiresAt) || payload.issuedAt > now + 30 || payload.expiresAt <= now ||
      payload.expiresAt - payload.issuedAt !== SESSION_DURATION || typeof payload.nonce !== 'string' || !/^[a-f0-9]{32}$/.test(payload.nonce)) return null;
    return payload;
  } catch {
    return null;
  }
}

export async function getSession() {
  const cookieStore = await cookies();
  return verifySessionToken(cookieStore.get(SESSION_COOKIE)?.value);
}

export async function setSession(username: string, secure: boolean) {
  const { token, maxAge } = createSessionToken(username);
  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.RSKYVIEW_SECURE_COOKIE === 'true' || secure,
    path: '/',
    maxAge,
  });
}

export async function deleteSession() {
  const cookieStore = await cookies();
  cookieStore.delete(SESSION_COOKIE);
}

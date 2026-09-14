'use client';

import Image from 'next/image';
import { useRef, useState } from 'react';
import { ArrowRight, Eye, EyeOff, LoaderCircle, LockKeyhole, Pause, Play, ShieldCheck, UserRound } from 'lucide-react';
import LoginBackdrop from './LoginBackdrop';
import styles from './LoginScreen.module.css';

export default function LoginScreen({ nextPath }: { nextPath: string }) {
  const passwordRef = useRef<HTMLInputElement>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [paused, setPaused] = useState(false);
  const [scene, setScene] = useState(0);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    setError('');
    setPending(true);
    const form = new FormData(event.currentTarget);
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: form.get('username'), password: form.get('password') }),
        signal: AbortSignal.timeout(12000),
      });
      if (!response.ok) {
        setError(response.status === 429
          ? 'Too many attempts. Please wait a minute and try again.'
          : response.status === 401
            ? 'Incorrect username or password. Please try again.'
            : 'Unable to sign in right now. Please try again.');
        passwordRef.current?.focus();
        setPending(false);
        return;
      }
      window.location.replace(nextPath);
    } catch {
      setError('Cannot connect to the server. Please try again.');
      setPending(false);
    }
  }

  function toggleMotion() {
    setPaused(value => !value);
  }

  return (
    <main className={styles.page}>
      <div className={styles.backdrop} aria-hidden="true">
        <LoginBackdrop scene={scene} paused={paused} />
      </div>
      <div className={styles.tint} aria-hidden="true" />

      <header className={styles.header}>
        <div className={styles.brand}>
          <Image src="/logo.png" alt="RTC Technology" width={116} height={60} priority className={styles.logo} />
          <span className={styles.productName}>R-SkyView<span>RTC Technology</span></span>
        </div>
        <span className={styles.headerNote}><span /> VISION WITHOUT LIMITS · 360°</span>
      </header>

      <div className={styles.main}>
        <section className={styles.card} aria-labelledby="login-title">
          <div className={styles.cardTop}><span className={styles.accessIcon}><LockKeyhole size={21} strokeWidth={1.5} /></span><span>YOUR INTELLIGENT WORKSPACE</span></div>
          <h1 id="login-title">Welcome Back</h1>
          <p className={styles.subtitle}>Please sign in to your R-SkyView account.</p>

          <form className={styles.form} onSubmit={submit} aria-busy={pending}>
            <div className={styles.field}>
              <label htmlFor="login-username">Username</label>
              <div className={styles.inputWrap}>
                <UserRound size={18} strokeWidth={1.6} aria-hidden="true" />
                <input id="login-username" name="username" autoComplete="username" placeholder="Enter your username"
                  autoCapitalize="none" spellCheck={false} required maxLength={128} disabled={pending}
                  aria-invalid={Boolean(error)} aria-describedby={error ? 'login-error' : undefined} onChange={() => setError('')} />
              </div>
            </div>
            <div className={styles.field}>
              <label htmlFor="login-password">Password</label>
              <div className={styles.inputWrap}>
                <LockKeyhole size={18} strokeWidth={1.6} aria-hidden="true" />
                <input ref={passwordRef} id="login-password" name="password" type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password" placeholder="Enter your password" required maxLength={256} disabled={pending}
                  aria-invalid={Boolean(error)} aria-describedby={error ? 'login-error' : undefined} onChange={() => setError('')} />
                <button className={styles.reveal} type="button" onClick={() => setShowPassword(value => !value)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'} aria-pressed={showPassword} disabled={pending}>
                  {showPassword ? <EyeOff size={18} strokeWidth={1.6} /> : <Eye size={18} strokeWidth={1.6} />}
                </button>
              </div>
            </div>
            <div className={styles.message} aria-live="polite" aria-atomic="true">
              {error ? <p id="login-error" role="alert">{error}</p> : <span><ShieldCheck size={13} /> Your workspace. Your access.</span>}
            </div>
            <button className={styles.submit} type="submit" disabled={pending}>
              <span>{pending ? 'Signing in…' : 'Sign in to R-SkyView'}</span>
              {pending ? <LoaderCircle className={styles.spinner} size={18} /> : <ArrowRight size={18} />}
            </button>
          </form>

          <div className={styles.cardFooter}><span /> BUILT FOR A CONNECTED WORLD <span /></div>
          <div className={styles.company}>Powered by <strong>RTC Technology</strong></div>
        </section>
      </div>

      <footer className={styles.footer}>
        <div><strong>A CONNECTED PERSPECTIVE</strong>© {new Date().getFullYear()} RTC Technology · R-SkyView</div>
        <div className={styles.sceneControls} aria-label="Background scenes">
          {['Midnight', 'Aurora', 'Orbit'].map((name, index) => (
            <button key={name} type="button" className={styles.sceneButton} aria-pressed={scene === index} onClick={() => setScene(index)}><i />{name}</button>
          ))}
          <button type="button" className={styles.motionButton} onClick={toggleMotion}
            aria-label={paused ? 'Play background animation' : 'Pause background animation'} aria-pressed={paused}>
            {paused ? <Play size={14} /> : <Pause size={14} />}
          </button>
        </div>
      </footer>
    </main>
  );
}

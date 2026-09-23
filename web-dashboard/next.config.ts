import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*' // Proxy to Python backend
      }
    ]
  },
  // @ts-ignore
  allowedDevOrigins: [
    '192.168.0.84',
    '192.168.0.84:3000',
    '192.168.5.212',
    '192.168.5.212:3000',
    '192.168.5.104',
    '192.168.5.105',
    '192.168.5.107',
    '192.168.5.103',
    '192.168.1.6',
    '192.168.53.77',
    'localhost',
    '127.0.0.1'
  ]
};

export default nextConfig;

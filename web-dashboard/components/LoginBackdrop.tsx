'use client';

import { useEffect, useRef } from 'react';
import styles from './LoginScreen.module.css';

const palettes = [
  { sky: '#060e1d', grid: '#627ea5', stars: '#d3e2ff', height: 27 },
  { sky: '#071a24', grid: '#3d9fa7', stars: '#b7ffea', height: 38 },
  { sky: '#120f28', grid: '#8872c1', stars: '#e3d1ff', height: 22 },
];

export default function LoginBackdrop({ scene, paused }: { scene: number; paused: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef(scene);
  const synchronizeRef = useRef<(() => void) | null>(null);
  const pausedRef = useRef(paused);

  useEffect(() => { sceneRef.current = scene; synchronizeRef.current?.(); }, [scene]);
  useEffect(() => { pausedRef.current = paused; synchronizeRef.current?.(); }, [paused]);

  useEffect(() => {
    let disposed = false;
    let cleanup = () => {};
    void import('three').then(THREE => {
      const canvas = canvasRef.current;
      if (disposed || !canvas) return;
      let renderer: InstanceType<typeof THREE.WebGLRenderer>;
      try {
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' });
      } catch { return; }
      const world = new THREE.Scene();
      const sky = new THREE.Color(palettes[0].sky);
      world.fog = new THREE.Fog(palettes[0].sky, 95, 220);
      const camera = new THREE.PerspectiveCamera(62, 1, 1, 400);
      const gridGeometry = new THREE.PlaneGeometry(340, 340, 74, 74);
      gridGeometry.rotateX(-Math.PI / 2);
      const gridMaterial = new THREE.MeshBasicMaterial({ color: palettes[0].grid, wireframe: true, transparent: true, opacity: .4 });
      const grid = new THREE.Mesh(gridGeometry, gridMaterial);
      world.add(grid);
      const positions = gridGeometry.attributes.position;
      const starsGeometry = new THREE.BufferGeometry();
      const stars = new Float32Array(650 * 3);
      for (let index = 0; index < 650; index++) {
        const angle = index * 2.399963;
        const radius = 90 + ((index * 73) % 110);
        stars[index * 3] = Math.cos(angle) * radius;
        stars[index * 3 + 1] = 10 + ((index * 37) % 150);
        stars[index * 3 + 2] = Math.sin(angle) * radius;
      }
      starsGeometry.setAttribute('position', new THREE.BufferAttribute(stars, 3));
      const starsMaterial = new THREE.PointsMaterial({ color: palettes[0].stars, size: .4, transparent: true, opacity: .7 });
      world.add(new THREE.Points(starsGeometry, starsMaterial));
      const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
      let animation = 0;
      let elapsed = 0;
      let previous = 0;
      let elevation = palettes[0].height;
      const targetColor = new THREE.Color();
      const running = () => !disposed && !document.hidden && !preference.matches && !pausedRef.current;

      const draw = (timestamp: number) => {
        animation = 0;
        if (disposed || document.hidden) return;
        if (running()) animation = requestAnimationFrame(draw);
        if (previous && timestamp - previous < 1000 / 30) return;
        const delta = previous ? Math.min((timestamp - previous) / 1000, .08) : 0;
        previous = timestamp;
        if (running()) elapsed += delta;
        const palette = palettes[sceneRef.current];
        const blend = running() ? .045 : 1;
        sky.lerp(targetColor.set(palette.sky), blend);
        renderer.setClearColor(sky);
        world.fog!.color.copy(sky);
        gridMaterial.color.lerp(targetColor.set(palette.grid), blend);
        starsMaterial.color.lerp(targetColor.set(palette.stars), blend);
        elevation += (palette.height - elevation) * blend;
        for (let index = 0; index < positions.count; index++) {
          const horizontal = positions.getX(index);
          const depth = positions.getZ(index);
          positions.setY(index, Math.sin(horizontal * .052 + elapsed * .24) * 5 + Math.cos(depth * .06 + elapsed * .16) * 5 + Math.sin((horizontal + depth) * .034) * 7);
        }
        positions.needsUpdate = true;
        const angle = elapsed * Math.PI / 90;
        camera.position.set(Math.sin(angle) * 78, elevation, Math.cos(angle) * 78);
        camera.lookAt(0, 20, 0);
        renderer.render(world, camera);
      };
      const synchronize = () => {
        cancelAnimationFrame(animation);
        previous = 0;
        if (!document.hidden) draw(performance.now());
      };
      const resize = () => {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        synchronize();
      };
      const observer = new ResizeObserver(resize);
      observer.observe(canvas);
      document.addEventListener('visibilitychange', synchronize);
      preference.addEventListener('change', synchronize);
      synchronizeRef.current = synchronize;
      resize();
      cleanup = () => {
        synchronizeRef.current = null;
        cancelAnimationFrame(animation);
        observer.disconnect();
        document.removeEventListener('visibilitychange', synchronize);
        preference.removeEventListener('change', synchronize);
        gridGeometry.dispose();
        gridMaterial.dispose();
        starsGeometry.dispose();
        starsMaterial.dispose();
        renderer.dispose();
        renderer.forceContextLoss();
      };
    }).catch(() => {});
    return () => { disposed = true; cleanup(); };
  }, []);

  return <canvas ref={canvasRef} className={styles.sceneCanvas} aria-hidden="true" />;
}

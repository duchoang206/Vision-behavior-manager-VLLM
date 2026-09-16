import * as THREE from 'three';

type FloorPoint = [number, number];
type FloorRect = [number, number, number, number];

export interface FactoryLayout {
  size?: { width?: number; depth?: number };
  grid?: { cell_size?: number };
  slam_walls?: [FloorPoint, FloorPoint][];
  walkways?: { polygon: FloorPoint[] }[];
  racks?: { id: string; rect: FloorRect }[];
  stations?: { id: string; rect: FloorRect }[];
  obstacles?: { id: string; rect: FloorRect }[];
  charging_stations?: { id: string; position?: [number, number, number]; access_point?: FloorPoint }[];
  locations?: { id: string; access_point?: FloorPoint; position?: [number, number, number] }[];
}

export function getFactoryView(layout: FactoryLayout | null) {
  const points = (layout?.walkways ?? []).flatMap(road => road.polygon).filter(point => point.every(Number.isFinite));
  if (!points.length) {
    const width = layout?.size?.width ?? 26;
    const depth = layout?.size?.depth ?? 18;
    return { centerX: width / 2, centerZ: depth / 2, span: Math.max(width, depth) };
  }
  const horizontal = points.map(point => point[0]);
  const vertical = points.map(point => point[1]);
  const minX = Math.min(...horizontal);
  const maxX = Math.max(...horizontal);
  const minZ = Math.min(...vertical);
  const maxZ = Math.max(...vertical);
  return { centerX: (minX + maxX) / 2, centerZ: (minZ + maxZ) / 2,
    span: Math.max(12, maxX - minX + 7, (maxZ - minZ + 5) * 1.4) };
}

export function disposeTwinObject(root: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  root.traverse(object => {
    const drawable = object as THREE.Mesh;
    if (drawable.geometry) geometries.add(drawable.geometry);
    if (drawable.material) {
      for (const material of Array.isArray(drawable.material) ? drawable.material : [drawable.material]) {
        materials.add(material);
        for (const value of Object.values(material)) {
          if (value instanceof THREE.Texture) textures.add(value);
        }
      }
    }
    if (object instanceof THREE.InstancedMesh) object.dispose();
  });
  textures.forEach(texture => texture.dispose());
  materials.forEach(material => material.dispose());
  geometries.forEach(geometry => geometry.dispose());
}

export function createFactoryFloor(layout: FactoryLayout | null) {
  const group = new THREE.Group();
  group.name = 'fms-factory-floor';
  const width = layout?.size?.width ?? 26;
  const depth = layout?.size?.depth ?? 18;
  const cell = Math.max(0.25, layout?.grid?.cell_size ?? 0.5);
  const steel = new THREE.MeshStandardMaterial({ color: 0x173640, roughness: 0.7, metalness: 0.4 });
  const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
  const cyan = new THREE.MeshBasicMaterial({ color: 0x4cebdd });
  const amber = new THREE.MeshBasicMaterial({ color: 0xf6c76a });
  const addBox = (dimensions: [number, number, number], position: [number, number, number], material: THREE.Material) => {
    const mesh = new THREE.Mesh(boxGeometry, material);
    mesh.scale.set(...dimensions);
    mesh.position.set(...position);
    group.add(mesh);
    return mesh;
  };
  const addLines = (points: number[], color: number, opacity: number) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity })));
  };
  addBox([width + 0.25, 0.24, depth + 0.25], [width / 2, -0.15, depth / 2], steel);
  addBox([width, 0.03, depth], [width / 2, -0.016, depth / 2],
    new THREE.MeshStandardMaterial({ color: 0x0b1d24, roughness: 0.82, metalness: 0.25 }));

  const grid: number[] = [];
  for (let position = 0; position <= width; position += cell) grid.push(position, 0.004, 0, position, 0.004, depth);
  for (let position = 0; position <= depth; position += cell) grid.push(0, 0.004, position, width, 0.004, position);
  addLines(grid, 0x3c7e85, 0.32);
  addLines([0, 0, 0, width, 0, 0, width, 0, 0, width, 0, depth,
    width, 0, depth, 0, 0, depth, 0, 0, depth, 0, 0, 0], 0x68d2cd, 0.7);

  const routeCore = new THREE.MeshBasicMaterial({ color: 0x4cf2e4 });
  const routeGlow = new THREE.MeshBasicMaterial({ color: 0x16d8d4, transparent: true, opacity: 0.12, depthWrite: false });
  const roadSurface = new THREE.MeshStandardMaterial({ color: 0x122f35, roughness: 0.85 });
  const chargers = (layout?.charging_stations ?? []).flatMap(item => {
    const point = item.access_point ?? (item.position ? [item.position[0], item.position[2]] as FloorPoint : null);
    return point ? [{ ...item, point }] : [];
  });
  const strip = (start: FloorPoint, end: FloorPoint, thickness: number, height: number, material: THREE.Material) => {
    const length = Math.hypot(end[0] - start[0], end[1] - start[1]);
    if (length < 0.01) return;
    const mesh = addBox([length, 0.005, thickness], [(start[0] + end[0]) / 2, height, (start[1] + end[1]) / 2], material);
    mesh.rotation.y = -Math.atan2(end[1] - start[1], end[0] - start[0]);
  };
  for (const road of layout?.walkways ?? []) {
    if (road.polygon.length !== 4) {
      road.polygon.forEach((point, index) => strip(point, road.polygon[(index + 1) % road.polygon.length], 0.025, 0.02, routeCore));
      continue;
    }
    const [first, second, third, fourth] = road.polygon;
    const shortFirst = Math.hypot(first[0] - second[0], first[1] - second[1]) < Math.hypot(second[0] - third[0], second[1] - third[1]);
    const start: FloorPoint = shortFirst ? [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2] : [(first[0] + fourth[0]) / 2, (first[1] + fourth[1]) / 2];
    const end: FloorPoint = shortFirst ? [(third[0] + fourth[0]) / 2, (third[1] + fourth[1]) / 2] : [(second[0] + third[0]) / 2, (second[1] + third[1]) / 2];
    const chargingRoad = chargers.some(({ point }) => [start, end].some(endpoint => Math.hypot(endpoint[0] - point[0], endpoint[1] - point[1]) < 0.15));
    strip(start, end, 0.64, 0.008, roadSurface);
    strip(start, end, 0.26, 0.015, routeGlow);
    strip(start, end, 0.035, 0.025, chargingRoad ? amber : routeCore);
  }

  const segments = (layout?.slam_walls ?? []).filter(([start, end]) =>
    [...start, ...end].every(Number.isFinite) && Math.hypot(end[0] - start[0], end[1] - start[1]) > 0.008);
  if (segments.length) {
    const walls = new THREE.InstancedMesh(boxGeometry, steel, segments.length);
    walls.name = 'fms-slam-structures';
    const matrix = new THREE.Object3D();
    const edges: number[] = [];
    segments.forEach(([start, end], index) => {
      const length = Math.hypot(end[0] - start[0], end[1] - start[1]);
      matrix.position.set((start[0] + end[0]) / 2, 0.54, (start[1] + end[1]) / 2);
      matrix.scale.set(length, 1.08, 0.045);
      matrix.rotation.y = -Math.atan2(end[1] - start[1], end[0] - start[0]);
      matrix.updateMatrix();
      walls.setMatrixAt(index, matrix.matrix);
      edges.push(start[0], 1.085, start[1], end[0], 1.085, end[1]);
    });
    walls.instanceMatrix.needsUpdate = true;
    group.add(walls);
    addLines(edges, 0x579496, 0.65);
  }

  const markerGeometry = new THREE.RingGeometry(0.055, 0.09, 16);
  for (const location of layout?.locations ?? []) {
    const point = location.access_point ?? (location.position ? [location.position[0], location.position[2]] : null);
    if (!point) continue;
    const marker = new THREE.Mesh(markerGeometry, cyan);
    marker.rotation.x = -Math.PI / 2;
    marker.position.set(point[0], 0.038, point[1]);
    group.add(marker);
  }
  const outline = (rect: FloorRect, material: THREE.Material) => {
    const [left, top, right, bottom] = rect;
    strip([left, top], [right, top], 0.025, 0.035, material);
    strip([right, top], [right, bottom], 0.025, 0.035, material);
    strip([right, bottom], [left, bottom], 0.025, 0.035, material);
    strip([left, bottom], [left, top], 0.025, 0.035, material);
  };
  for (const station of layout?.stations ?? []) outline(station.rect, cyan);
  for (const { point } of chargers) {
    const [horizontal, vertical] = point;
    outline([horizontal - 0.38, vertical - 0.38, horizontal + 0.38, vertical + 0.38], amber);
    addBox([0.3, 0.65, 0.12], [horizontal, 0.325, vertical - 0.28], steel);
    addBox([0.19, 0.16, 0.02], [horizontal, 0.46, vertical - 0.21], amber);
  }
  const cargoMaterial = new THREE.MeshStandardMaterial({ color: 0x6d7265, roughness: 0.9 });
  for (const rack of layout?.racks ?? []) {
    const [left, top, right, bottom] = rack.rect;
    for (const horizontal of [left, right]) {
      for (const vertical of [top, bottom]) addBox([0.055, 2.1, 0.055], [horizontal, 1.05, vertical], steel);
    }
    for (let level = 0; level < 3; level++) {
      addBox([right - left, 0.055, bottom - top], [(left + right) / 2, 0.22 + level * 0.66, (top + bottom) / 2], steel);
      addBox([(right - left) * 0.84, 0.42, (bottom - top) * 0.78], [(left + right) / 2, 0.46 + level * 0.66, (top + bottom) / 2], cargoMaterial);
    }
    outline(rack.rect, cyan);
  }
  for (const obstacle of layout?.obstacles ?? []) {
    const [left, top, right, bottom] = obstacle.rect;
    addBox([right - left, 0.6, bottom - top], [(left + right) / 2, 0.3, (top + bottom) / 2], steel);
  }
  return group;
}

export function createRobotTrail() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(96 * 3), 3).setUsage(THREE.DynamicDrawUsage));
  geometry.setDrawRange(0, 0);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x48eadd, transparent: true, opacity: 0.65, depthWrite: false }));
  line.frustumCulled = false;
  return { line, points: [] as { position: [number, number, number]; at: number }[] };
}

export function updateRobotTrail(trail: ReturnType<typeof createRobotTrail>, position: [number, number, number], timestamp: number, color: number) {
  const previous = trail.points.at(-1);
  const distance = previous ? Math.hypot(previous.position[0] - position[0], previous.position[2] - position[2]) : 0;
  if (distance > 2.5 || (previous && timestamp - previous.at > 5000)) trail.points.length = 0;
  if (!trail.points.length || distance >= 0.06) trail.points.push({ position: [...position], at: timestamp });
  trail.points = trail.points.filter(point => timestamp - point.at < 16000).slice(-96);
  const attribute = trail.line.geometry.getAttribute('position') as THREE.BufferAttribute;
  trail.points.forEach((point, index) => attribute.setXYZ(index, point.position[0], 0.055, point.position[2]));
  attribute.needsUpdate = true;
  trail.line.geometry.setDrawRange(0, trail.points.length);
  trail.line.material.color.set(color);
}

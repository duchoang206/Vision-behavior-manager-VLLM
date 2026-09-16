export type Point2 = [number, number];
export type CalibrationPair = { camera: Point2; floor: Point2 };
export type FmsFrame = { origin_x: number; origin_y: number; layout_depth: number };

export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): Point2 | null {
  const matrix = svg.getScreenCTM();
  if (!matrix) return null;
  const point = new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
  return Number.isFinite(point.x) && Number.isFinite(point.y) ? [point.x, point.y] : null;
}

export function floorToFms(point: Point2, frame: FmsFrame): Point2 {
  return [point[0] + frame.origin_x, frame.origin_y + frame.layout_depth - point[1]];
}

export function isDistinctPair(pairs: CalibrationPair[], camera: Point2, floor: Point2) {
  return pairs.every(pair => Math.hypot(pair.camera[0] - camera[0], pair.camera[1] - camera[1]) > 0.0001
    && Math.hypot(pair.floor[0] - floor[0], pair.floor[1] - floor[1]) > 0.001);
}

// Physics simulation engine

import type { Vector2, PhysicsState, ForceField, WindField, VortexField, PointForce } from './types';

export const SIMULATION_CONFIG = {
  dt: 0.016, // 60fps
  gravity: 0,
  damping: 0.99,
  bounds: { width: 800, height: 600 },
};

// Vector math utilities
export const vec2 = {
  add: (a: Vector2, b: Vector2): Vector2 => ({ x: a.x + b.x, y: a.y + b.y }),
  sub: (a: Vector2, b: Vector2): Vector2 => ({ x: a.x - b.x, y: a.y - b.y }),
  mul: (v: Vector2, s: number): Vector2 => ({ x: v.x * s, y: v.y * s }),
  div: (v: Vector2, s: number): Vector2 => ({ x: v.x / s, y: v.y / s }),
  dot: (a: Vector2, b: Vector2): number => a.x * b.x + a.y * b.y,
  length: (v: Vector2): number => Math.sqrt(v.x * v.x + v.y * v.y),
  normalize: (v: Vector2): Vector2 => {
    const len = vec2.length(v);
    return len > 0 ? vec2.div(v, len) : { x: 0, y: 0 };
  },
  rotate: (v: Vector2, angle: number): Vector2 => ({
    x: v.x * Math.cos(angle) - v.y * Math.sin(angle),
    y: v.x * Math.sin(angle) + v.y * Math.cos(angle),
  }),
  perpendicular: (v: Vector2): Vector2 => ({ x: -v.y, y: v.x }),
  distance: (a: Vector2, b: Vector2): number => vec2.length(vec2.sub(b, a)),
};

// Soft indicator functions for smooth force transitions
function softIndicatorBox(pos: Vector2, center: Vector2, size: Vector2, softness: number = 0.1): number {
  const relX = (pos.x - center.x) / size.x;
  const relY = (pos.y - center.y) / size.y;
  const scale = 1.0 / softness;
  const indX = 1.0 / (1.0 + Math.exp(scale * (Math.abs(relX) - 1.0)));
  const indY = 1.0 / (1.0 + Math.exp(scale * (Math.abs(relY) - 1.0)));
  return indX * indY;
}

function softIndicatorCircle(pos: Vector2, center: Vector2, radius: number, softness: number = 0.1): number {
  const dist = vec2.distance(pos, center);
  const relDist = dist / radius;
  const scale = 1.0 / softness;
  return 1.0 / (1.0 + Math.exp(scale * (relDist - 1.0)));
}

// Force field calculations
function computeWindForce(field: WindField, pos: Vector2): Vector2 {
  const indicator = softIndicatorBox(pos, field.center, field.size);
  return vec2.mul(field.direction, field.strength * indicator);
}

function computeVortexForce(field: VortexField, pos: Vector2): Vector2 {
  const indicator = softIndicatorCircle(pos, field.center, field.radius);
  const toCenter = vec2.sub(pos, field.center);
  const tangent = vec2.perpendicular(vec2.normalize(toCenter));
  return vec2.mul(tangent, field.strength * indicator);
}

function computePointForce(field: PointForce, pos: Vector2): Vector2 {
  const indicator = softIndicatorCircle(pos, field.center, field.radius);
  const toCenter = vec2.sub(field.center, pos);
  const dist = vec2.length(toCenter);
  if (dist < 0.1) return { x: 0, y: 0 };
  
  const direction = vec2.normalize(toCenter);
  const sign = field.type === 'attractor' ? 1 : -1;
  const falloff = 1.0 / (1.0 + dist * 0.1);
  
  return vec2.mul(direction, sign * field.strength * indicator * falloff);
}

export function computeTotalForce(fields: ForceField[], pos: Vector2): Vector2 {
  let force: Vector2 = { x: 0, y: 0 };
  
  for (const field of fields) {
    let fieldForce: Vector2;
    
    switch (field.type) {
      case 'wind':
        fieldForce = computeWindForce(field as WindField, pos);
        break;
      case 'vortex':
        fieldForce = computeVortexForce(field as VortexField, pos);
        break;
      case 'attractor':
      case 'repeller':
        fieldForce = computePointForce(field as PointForce, pos);
        break;
      default:
        fieldForce = { x: 0, y: 0 };
    }
    
    force = vec2.add(force, fieldForce);
  }
  
  return force;
}

export function simulateStep(state: PhysicsState, fields: ForceField[], dt: number): PhysicsState {
  const force = computeTotalForce(fields, state.position);
  
  // Semi-implicit Euler integration
  const newVelocity = vec2.add(state.velocity, vec2.mul(force, dt));
  const dampedVelocity = vec2.mul(newVelocity, SIMULATION_CONFIG.damping);
  const newPosition = vec2.add(state.position, vec2.mul(dampedVelocity, dt));
  
  // Bounce off walls
  let finalPos = { ...newPosition };
  let finalVel = { ...dampedVelocity };
  
  const { width, height } = SIMULATION_CONFIG.bounds;
  const margin = 10;
  
  if (finalPos.x < margin) {
    finalPos.x = margin;
    finalVel.x = -finalVel.x * 0.8;
  } else if (finalPos.x > width - margin) {
    finalPos.x = width - margin;
    finalVel.x = -finalVel.x * 0.8;
  }
  
  if (finalPos.y < margin) {
    finalPos.y = margin;
    finalVel.y = -finalVel.y * 0.8;
  } else if (finalPos.y > height - margin) {
    finalPos.y = height - margin;
    finalVel.y = -finalVel.y * 0.8;
  }
  
  return {
    position: finalPos,
    velocity: finalVel,
    time: state.time + dt,
  };
}

export function simulateTrajectory(
  initialState: PhysicsState,
  fields: ForceField[],
  maxSteps: number = 500
): Vector2[] {
  const trajectory: Vector2[] = [{ ...initialState.position }];
  let state = { ...initialState };
  
  for (let i = 0; i < maxSteps; i++) {
    state = simulateStep(state, fields, SIMULATION_CONFIG.dt);
    trajectory.push({ ...state.position });
    
    // Stop if velocity is very low (ball stopped)
    if (vec2.length(state.velocity) < 0.5) {
      break;
    }
  }
  
  return trajectory;
}

export function checkWinCondition(
  position: Vector2,
  targetPosition: Vector2,
  targetRadius: number
): boolean {
  return vec2.distance(position, targetPosition) < targetRadius;
}


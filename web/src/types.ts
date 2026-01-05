// Game types

export interface Vector2 {
  x: number;
  y: number;
}

export interface PhysicsState {
  position: Vector2;
  velocity: Vector2;
  time: number;
}

export interface WindField {
  id: string;
  type: 'wind';
  center: Vector2;
  size: Vector2;
  direction: Vector2;
  strength: number;
  color: string;
}

export interface VortexField {
  id: string;
  type: 'vortex';
  center: Vector2;
  radius: number;
  strength: number;
  color: string;
}

export interface PointForce {
  id: string;
  type: 'attractor' | 'repeller';
  center: Vector2;
  strength: number;
  radius: number;
  color: string;
}

export type ForceField = WindField | VortexField | PointForce;

export interface Level {
  id: number;
  name: string;
  description: string;
  startPosition: Vector2;
  startVelocity: Vector2;
  targetPosition: Vector2;
  targetRadius: number;
  obstacles: Obstacle[];
  availableFields: FieldType[];
  maxFields: number;
  parTime: number;
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
}

export interface Obstacle {
  type: 'rectangle' | 'circle';
  position: Vector2;
  size?: Vector2;
  radius?: number;
}

export type FieldType = 'wind' | 'vortex' | 'attractor' | 'repeller';

export interface GameState {
  currentLevel: number;
  isPlaying: boolean;
  isWon: boolean;
  ball: PhysicsState;
  trajectory: Vector2[];
  placedFields: ForceField[];
  selectedFieldType: FieldType | null;
  time: number;
}





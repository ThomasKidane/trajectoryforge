// Game state management with Zustand

import { create } from 'zustand';
import type { GameState, ForceField, FieldType, Vector2, PhysicsState } from './types';
import { getLevel, LEVELS } from './levels';
import { simulateStep, checkWinCondition, SIMULATION_CONFIG, vec2 } from './physics';

interface GameStore extends GameState {
  // Actions
  setLevel: (levelId: number) => void;
  startSimulation: () => void;
  stopSimulation: () => void;
  resetLevel: () => void;
  tick: () => void;
  addField: (field: ForceField) => void;
  removeField: (fieldId: string) => void;
  updateField: (fieldId: string, updates: Partial<ForceField>) => void;
  selectFieldType: (type: FieldType | null) => void;
  clearFields: () => void;
}

const createInitialBallState = (levelId: number): PhysicsState => {
  const level = getLevel(levelId);
  if (!level) {
    return { position: { x: 100, y: 300 }, velocity: { x: 0, y: 0 }, time: 0 };
  }
  return {
    position: { ...level.startPosition },
    velocity: { ...level.startVelocity },
    time: 0,
  };
};

export const useGameStore = create<GameStore>((set, get) => ({
  // Initial state
  currentLevel: 1,
  isPlaying: false,
  isWon: false,
  ball: createInitialBallState(1),
  trajectory: [],
  placedFields: [],
  selectedFieldType: null,
  time: 0,

  // Actions
  setLevel: (levelId: number) => {
    const level = getLevel(levelId);
    if (level) {
      set({
        currentLevel: levelId,
        isPlaying: false,
        isWon: false,
        ball: createInitialBallState(levelId),
        trajectory: [],
        placedFields: [],
        selectedFieldType: null,
        time: 0,
      });
    }
  },

  startSimulation: () => {
    const state = get();
    set({
      isPlaying: true,
      isWon: false,
      ball: createInitialBallState(state.currentLevel),
      trajectory: [{ ...createInitialBallState(state.currentLevel).position }],
      time: 0,
    });
  },

  stopSimulation: () => {
    set({ isPlaying: false });
  },

  resetLevel: () => {
    const state = get();
    set({
      isPlaying: false,
      isWon: false,
      ball: createInitialBallState(state.currentLevel),
      trajectory: [],
      time: 0,
    });
  },

  tick: () => {
    const state = get();
    if (!state.isPlaying || state.isWon) return;

    const level = getLevel(state.currentLevel);
    if (!level) return;

    // Simulate one step
    const newBall = simulateStep(state.ball, state.placedFields, SIMULATION_CONFIG.dt);
    const newTrajectory = [...state.trajectory, { ...newBall.position }];

    // Check win condition
    const won = checkWinCondition(newBall.position, level.targetPosition, level.targetRadius);

    // Check if ball stopped (lost)
    const stopped = vec2.length(newBall.velocity) < 0.5 && state.time > 1;

    set({
      ball: newBall,
      trajectory: newTrajectory,
      time: state.time + SIMULATION_CONFIG.dt,
      isWon: won,
      isPlaying: won || stopped ? false : true,
    });
  },

  addField: (field: ForceField) => {
    const state = get();
    const level = getLevel(state.currentLevel);
    if (!level) return;

    if (state.placedFields.length >= level.maxFields) return;

    set({ placedFields: [...state.placedFields, field] });
  },

  removeField: (fieldId: string) => {
    const state = get();
    set({ placedFields: state.placedFields.filter(f => f.id !== fieldId) });
  },

  updateField: (fieldId: string, updates: Partial<ForceField>) => {
    const state = get();
    set({
      placedFields: state.placedFields.map(f =>
        f.id === fieldId ? { ...f, ...updates } as ForceField : f
      ),
    });
  },

  selectFieldType: (type: FieldType | null) => {
    set({ selectedFieldType: type });
  },

  clearFields: () => {
    set({ placedFields: [] });
  },
}));


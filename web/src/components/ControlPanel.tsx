// Control Panel Component

import { useGameStore } from '../store';
import { getLevel, getNextLevel, LEVELS } from '../levels';
import type { FieldType } from '../types';
import { motion } from 'framer-motion';

const FIELD_INFO: Record<FieldType, { name: string; icon: string; description: string; color: string }> = {
  wind: {
    name: 'Wind',
    icon: '💨',
    description: 'Pushes the ball in one direction',
    color: '#40e0d0',
  },
  vortex: {
    name: 'Vortex',
    icon: '🌀',
    description: 'Spins the ball around',
    color: '#8a2be2',
  },
  attractor: {
    name: 'Attractor',
    icon: '🧲',
    description: 'Pulls the ball toward it',
    color: '#32cd32',
  },
  repeller: {
    name: 'Repeller',
    icon: '⚡',
    description: 'Pushes the ball away',
    color: '#ff4500',
  },
};

export function ControlPanel() {
  const {
    currentLevel,
    isPlaying,
    isWon,
    placedFields,
    selectedFieldType,
    time,
    setLevel,
    startSimulation,
    stopSimulation,
    resetLevel,
    selectFieldType,
    clearFields,
    removeField,
  } = useGameStore();

  const level = getLevel(currentLevel);
  const nextLevel = getNextLevel(currentLevel);

  if (!level) return null;

  return (
    <div className="control-panel">
      {/* Level Info */}
      <div className="panel-section">
        <h2 className="level-title">
          Level {level.id}: {level.name}
        </h2>
        <p className="level-description">{level.description}</p>
        <div className="level-meta">
          <span className={`difficulty ${level.difficulty}`}>
            {level.difficulty.toUpperCase()}
          </span>
          <span className="par-time">Par: {level.parTime}s</span>
        </div>
      </div>

      {/* Stats */}
      <div className="panel-section stats">
        <div className="stat">
          <span className="stat-label">Time</span>
          <span className="stat-value">{time.toFixed(1)}s</span>
        </div>
        <div className="stat">
          <span className="stat-label">Fields</span>
          <span className="stat-value">{placedFields.length}/{level.maxFields}</span>
        </div>
      </div>

      {/* Field Selection */}
      <div className="panel-section">
        <h3>Force Fields</h3>
        <div className="field-buttons">
          {level.availableFields.map((type) => {
            const info = FIELD_INFO[type];
            const isSelected = selectedFieldType === type;
            const isDisabled = isPlaying || placedFields.length >= level.maxFields;
            
            return (
              <motion.button
                key={type}
                className={`field-button ${isSelected ? 'selected' : ''}`}
                onClick={() => selectFieldType(isSelected ? null : type)}
                disabled={isDisabled}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{
                  borderColor: isSelected ? info.color : 'transparent',
                  backgroundColor: isSelected ? `${info.color}20` : undefined,
                }}
              >
                <span className="field-icon">{info.icon}</span>
                <span className="field-name">{info.name}</span>
              </motion.button>
            );
          })}
        </div>
        {selectedFieldType && (
          <p className="field-hint">
            Click on the canvas to place a {FIELD_INFO[selectedFieldType].name} field
          </p>
        )}
      </div>

      {/* Placed Fields */}
      {placedFields.length > 0 && (
        <div className="panel-section">
          <h3>Placed Fields</h3>
          <div className="placed-fields">
            {placedFields.map((field, index) => (
              <div key={field.id} className="placed-field">
                <span>{FIELD_INFO[field.type as FieldType]?.icon} {field.type}</span>
                <button
                  className="remove-field"
                  onClick={() => removeField(field.id)}
                  disabled={isPlaying}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="panel-section controls">
        {!isPlaying && !isWon && (
          <>
            <motion.button
              className="btn btn-primary"
              onClick={startSimulation}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              ▶ Launch Ball
            </motion.button>
            <motion.button
              className="btn btn-secondary"
              onClick={clearFields}
              disabled={placedFields.length === 0}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Clear Fields
            </motion.button>
          </>
        )}
        
        {isPlaying && (
          <motion.button
            className="btn btn-danger"
            onClick={stopSimulation}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            ⏹ Stop
          </motion.button>
        )}
        
        {isWon && (
          <>
            <div className="win-message">
              🎉 Completed in {time.toFixed(1)}s!
              {time <= level.parTime && <span className="par-beat"> ⭐ Under par!</span>}
            </div>
            {nextLevel ? (
              <motion.button
                className="btn btn-success"
                onClick={() => setLevel(nextLevel.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Next Level →
              </motion.button>
            ) : (
              <div className="all-complete">🏆 All levels complete!</div>
            )}
          </>
        )}
        
        {(isPlaying || isWon) && (
          <motion.button
            className="btn btn-secondary"
            onClick={resetLevel}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            ↺ Reset
          </motion.button>
        )}
      </div>

      {/* Level Select */}
      <div className="panel-section level-select">
        <h3>Levels</h3>
        <div className="level-grid">
          {LEVELS.map((l) => (
            <motion.button
              key={l.id}
              className={`level-btn ${currentLevel === l.id ? 'current' : ''} ${l.difficulty}`}
              onClick={() => setLevel(l.id)}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              {l.id}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}


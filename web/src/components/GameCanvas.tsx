// Game Canvas Component

import { useEffect, useRef, useCallback } from 'react';
import { useGameStore } from '../store';
import { getLevel } from '../levels';
import type { ForceField, WindField, VortexField, PointForce, Vector2 } from '../types';
import { SIMULATION_CONFIG, simulateTrajectory } from '../physics';

const COLORS = {
  background: '#0a0a0f',
  grid: '#1a1a2e',
  ball: '#00ff88',
  ballGlow: 'rgba(0, 255, 136, 0.3)',
  target: '#ff6b6b',
  targetGlow: 'rgba(255, 107, 107, 0.3)',
  trajectory: 'rgba(0, 255, 136, 0.4)',
  trajectoryPreview: 'rgba(255, 255, 255, 0.2)',
  obstacle: '#2d2d44',
  wind: 'rgba(64, 224, 208, 0.3)',
  windBorder: '#40e0d0',
  vortex: 'rgba(138, 43, 226, 0.3)',
  vortexBorder: '#8a2be2',
  attractor: 'rgba(50, 205, 50, 0.3)',
  attractorBorder: '#32cd32',
  repeller: 'rgba(255, 69, 0, 0.3)',
  repellerBorder: '#ff4500',
};

export function GameCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  const {
    currentLevel,
    isPlaying,
    isWon,
    ball,
    trajectory,
    placedFields,
    selectedFieldType,
    tick,
    addField,
  } = useGameStore();

  const level = getLevel(currentLevel);

  // Handle canvas click to place fields
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPlaying || !selectedFieldType || !level) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const id = `field-${Date.now()}`;
    let newField: ForceField;

    switch (selectedFieldType) {
      case 'wind':
        newField = {
          id,
          type: 'wind',
          center: { x, y },
          size: { x: 80, y: 60 },
          direction: { x: 1, y: 0 },
          strength: 200,
          color: COLORS.windBorder,
        } as WindField;
        break;
      case 'vortex':
        newField = {
          id,
          type: 'vortex',
          center: { x, y },
          radius: 60,
          strength: 150,
          color: COLORS.vortexBorder,
        } as VortexField;
        break;
      case 'attractor':
        newField = {
          id,
          type: 'attractor',
          center: { x, y },
          strength: 300,
          radius: 100,
          color: COLORS.attractorBorder,
        } as PointForce;
        break;
      case 'repeller':
        newField = {
          id,
          type: 'repeller',
          center: { x, y },
          strength: 300,
          radius: 100,
          color: COLORS.repellerBorder,
        } as PointForce;
        break;
      default:
        return;
    }

    addField(newField);
  }, [isPlaying, selectedFieldType, level, addField]);

  // Game loop
  useEffect(() => {
    if (isPlaying && !isWon) {
      const gameLoop = () => {
        tick();
        animationRef.current = requestAnimationFrame(gameLoop);
      };
      animationRef.current = requestAnimationFrame(gameLoop);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, isWon, tick]);

  // Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !level) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = SIMULATION_CONFIG.bounds;

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw obstacles
    ctx.fillStyle = COLORS.obstacle;
    for (const obs of level.obstacles) {
      if (obs.type === 'rectangle' && obs.size) {
        ctx.fillRect(obs.position.x, obs.position.y, obs.size.x, obs.size.y);
      } else if (obs.type === 'circle' && obs.radius) {
        ctx.beginPath();
        ctx.arc(obs.position.x, obs.position.y, obs.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw force fields
    for (const field of placedFields) {
      drawForceField(ctx, field);
    }

    // Draw trajectory preview (when not playing)
    if (!isPlaying && placedFields.length > 0) {
      const previewTrajectory = simulateTrajectory(
        {
          position: { ...level.startPosition },
          velocity: { ...level.startVelocity },
          time: 0,
        },
        placedFields,
        300
      );
      
      ctx.strokeStyle = COLORS.trajectoryPreview;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      previewTrajectory.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw actual trajectory
    if (trajectory.length > 1) {
      ctx.strokeStyle = COLORS.trajectory;
      ctx.lineWidth = 3;
      ctx.beginPath();
      trajectory.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
    }

    // Draw target
    ctx.beginPath();
    ctx.arc(level.targetPosition.x, level.targetPosition.y, level.targetRadius + 10, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.targetGlow;
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(level.targetPosition.x, level.targetPosition.y, level.targetRadius, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.target;
    ctx.fill();
    
    // Target inner ring
    ctx.beginPath();
    ctx.arc(level.targetPosition.x, level.targetPosition.y, level.targetRadius * 0.6, 0, Math.PI * 2);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw ball
    const ballPos = isPlaying ? ball.position : level.startPosition;
    
    // Ball glow
    ctx.beginPath();
    ctx.arc(ballPos.x, ballPos.y, 20, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.ballGlow;
    ctx.fill();
    
    // Ball
    ctx.beginPath();
    ctx.arc(ballPos.x, ballPos.y, 12, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.ball;
    ctx.fill();
    
    // Ball highlight
    ctx.beginPath();
    ctx.arc(ballPos.x - 3, ballPos.y - 3, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.fill();

    // Draw velocity indicator (when not playing)
    if (!isPlaying) {
      const velScale = 0.5;
      ctx.strokeStyle = COLORS.ball;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(level.startPosition.x, level.startPosition.y);
      ctx.lineTo(
        level.startPosition.x + level.startVelocity.x * velScale,
        level.startPosition.y + level.startVelocity.y * velScale
      );
      ctx.stroke();
      
      // Arrow head
      const angle = Math.atan2(level.startVelocity.y, level.startVelocity.x);
      const arrowSize = 8;
      const endX = level.startPosition.x + level.startVelocity.x * velScale;
      const endY = level.startPosition.y + level.startVelocity.y * velScale;
      
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - arrowSize * Math.cos(angle - Math.PI / 6),
        endY - arrowSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        endX - arrowSize * Math.cos(angle + Math.PI / 6),
        endY - arrowSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fillStyle = COLORS.ball;
      ctx.fill();
    }

    // Win overlay
    if (isWon) {
      ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
      ctx.fillRect(0, 0, width, height);
      
      ctx.font = 'bold 48px "Space Grotesk", sans-serif';
      ctx.fillStyle = COLORS.ball;
      ctx.textAlign = 'center';
      ctx.fillText('🎉 LEVEL COMPLETE!', width / 2, height / 2);
    }

  }, [currentLevel, isPlaying, isWon, ball, trajectory, placedFields, level]);

  return (
    <canvas
      ref={canvasRef}
      width={SIMULATION_CONFIG.bounds.width}
      height={SIMULATION_CONFIG.bounds.height}
      onClick={handleCanvasClick}
      style={{
        cursor: selectedFieldType && !isPlaying ? 'crosshair' : 'default',
        borderRadius: '12px',
        boxShadow: '0 0 40px rgba(0, 255, 136, 0.1)',
      }}
    />
  );
}

function drawForceField(ctx: CanvasRenderingContext2D, field: ForceField) {
  ctx.save();
  
  switch (field.type) {
    case 'wind': {
      const f = field as WindField;
      ctx.fillStyle = COLORS.wind;
      ctx.strokeStyle = COLORS.windBorder;
      ctx.lineWidth = 2;
      
      ctx.fillRect(
        f.center.x - f.size.x,
        f.center.y - f.size.y,
        f.size.x * 2,
        f.size.y * 2
      );
      ctx.strokeRect(
        f.center.x - f.size.x,
        f.center.y - f.size.y,
        f.size.x * 2,
        f.size.y * 2
      );
      
      // Draw wind direction arrows
      const arrowCount = 3;
      for (let i = 0; i < arrowCount; i++) {
        const offsetX = (i - 1) * f.size.x * 0.6;
        drawArrow(ctx, 
          f.center.x + offsetX - f.direction.x * 20,
          f.center.y - f.direction.y * 20,
          f.center.x + offsetX + f.direction.x * 20,
          f.center.y + f.direction.y * 20,
          COLORS.windBorder
        );
      }
      break;
    }
    
    case 'vortex': {
      const f = field as VortexField;
      ctx.fillStyle = COLORS.vortex;
      ctx.strokeStyle = COLORS.vortexBorder;
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      ctx.arc(f.center.x, f.center.y, f.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Draw spiral
      ctx.strokeStyle = COLORS.vortexBorder;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let angle = 0; angle < Math.PI * 4; angle += 0.1) {
        const r = (angle / (Math.PI * 4)) * f.radius * 0.8;
        const x = f.center.x + r * Math.cos(angle);
        const y = f.center.y + r * Math.sin(angle);
        if (angle === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      break;
    }
    
    case 'attractor':
    case 'repeller': {
      const f = field as PointForce;
      const isAttractor = field.type === 'attractor';
      ctx.fillStyle = isAttractor ? COLORS.attractor : COLORS.repeller;
      ctx.strokeStyle = isAttractor ? COLORS.attractorBorder : COLORS.repellerBorder;
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      ctx.arc(f.center.x, f.center.y, f.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Draw arrows pointing in/out
      const arrowAngles = [0, Math.PI / 2, Math.PI, Math.PI * 1.5];
      for (const angle of arrowAngles) {
        const innerR = 15;
        const outerR = f.radius * 0.6;
        
        if (isAttractor) {
          drawArrow(ctx,
            f.center.x + outerR * Math.cos(angle),
            f.center.y + outerR * Math.sin(angle),
            f.center.x + innerR * Math.cos(angle),
            f.center.y + innerR * Math.sin(angle),
            COLORS.attractorBorder
          );
        } else {
          drawArrow(ctx,
            f.center.x + innerR * Math.cos(angle),
            f.center.y + innerR * Math.sin(angle),
            f.center.x + outerR * Math.cos(angle),
            f.center.y + outerR * Math.sin(angle),
            COLORS.repellerBorder
          );
        }
      }
      break;
    }
  }
  
  ctx.restore();
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  color: string
) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const arrowSize = 6;
  
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();
  
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - arrowSize * Math.cos(angle - Math.PI / 6),
    toY - arrowSize * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    toX - arrowSize * Math.cos(angle + Math.PI / 6),
    toY - arrowSize * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}


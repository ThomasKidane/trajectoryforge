import { GameCanvas } from './components/GameCanvas';
import { ControlPanel } from './components/ControlPanel';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="header">
        <h1 className="title">
          <span className="title-icon">🎯</span>
          TrajectoryForge
        </h1>
        <p className="subtitle">Master the physics. Shape the path.</p>
      </header>
      
      <main className="game-container">
        <div className="canvas-wrapper">
          <GameCanvas />
        </div>
        <aside className="sidebar">
          <ControlPanel />
        </aside>
      </main>
      
      <footer className="footer">
        <p>Place force fields to guide the ball to the target 🎯</p>
      </footer>
    </div>
  );
}

export default App;

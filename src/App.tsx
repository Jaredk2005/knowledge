import React, { useState, useEffect } from 'react';
import { Dashboard } from './components/Dashboard';
import { LoginPage } from './components/LoginPage';
import { authService } from './services/authService';
import { User } from './types/user';
import { io } from 'socket.io-client';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [threats, setThreats] = useState<any[]>([]);

  useEffect(() => {
    // Check if user is already logged in
    const currentUser = authService.getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
      setIsAuthenticated(true);
    }
    setIsLoading(false);

    // Connect to backend via WebSocket
    const socket = io('http://localhost:5173'); // Change URL if backend is hosted elsewhere

    socket.on('connect', () => {
      console.log('Connected to threat detection backend');
    });

    socket.on('threat_detected', (data) => {
      console.log('Threat detected:', data);
      setThreats((prev) => [data, ...prev]); // Add newest threats at the top
    });

    socket.on('disconnect', () => {
      console.warn('Disconnected from backend');
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const handleLogin = (loggedInUser: User) => {
    setUser(loggedInUser);
    setIsAuthenticated(true);
  };

  const handleLogout = async () => {
    await authService.logout();
    setUser(null);
    setIsAuthenticated(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return <Dashboard user={user} threats={threats} onLogout={handleLogout} />;
}

export default App;

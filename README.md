# Cybersecurity Incident Response Automation Tool

A comprehensive cybersecurity dashboard for real-time threat monitoring, incident response automation, and security operations center (SOC) management.

## Features

- **Real-time Threat Detection**: Live monitoring of security incidents with automated classification
- **Email Alert System**: Python-based system monitoring with email notifications for resource thresholds
- **Network Traffic Analysis**: Continuous monitoring of network activity with suspicious behavior detection
- **Incident Management**: Comprehensive incident tracking with automated response actions
- **Alert System**: Real-time alerts with acknowledgment and escalation capabilities
- **System Status Monitoring**: Health monitoring of critical security infrastructure
- **Interactive Threat Map**: Visual representation of global threat landscape
- **Automated Response**: Configurable response playbooks for different incident types

## System Requirements

### Frontend Dashboard
## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Build Tool**: Vite
- **Linting**: ESLint + TypeScript ESLint

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- VS Code (recommended)

### Email Alert System
- Python 3.7+
- psutil library
- SMTP server access (Gmail, Outlook, etc.)

### Installation

#### Frontend Dashboard
1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

#### Email Alert System
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure email settings in `alert_config.json`

3. Test the system:
   ```bash
   python system_monitor.py --test
   ```

4. Start monitoring:
   ```bash
   python system_monitor.py --monitor
   ```

### VS Code Setup

This project includes VS Code configuration for optimal development experience:

- **Recommended Extensions**: Automatically suggests useful extensions
- **Settings**: Pre-configured for TypeScript, Tailwind CSS, and React development
- **Tasks**: Quick access to npm scripts via Command Palette
- **Debugging**: Chrome debugging configuration included

#### Recommended VS Code Extensions

- Tailwind CSS IntelliSense
- Prettier - Code formatter
- ESLint
- TypeScript and JavaScript Language Features
- Auto Rename Tag
- Path Intellisense

### Development Commands

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

### System Monitoring Commands

- `python system_monitor.py --test` - Send test email
- `python system_monitor.py --check` - Run single system check
- `python system_monitor.py --monitor` - Start continuous monitoring
- `python system_monitor.py --config custom_config.json` - Use custom config

### VS Code Usage

1. **Running the Development Server**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Tasks: Run Task"
   - Select "npm: dev"

2. **Debugging**:
   - Set breakpoints in your TypeScript/React code
   - Press `F5` or go to Run and Debug panel
   - Select "Launch Chrome" configuration

3. **Code Formatting**:
   - Files are automatically formatted on save
   - Manual format: `Shift+Alt+F` (or `Shift+Option+F` on Mac)

## Project Structure

```
src/
├── components/          # React components
│   ├── Dashboard.tsx    # Main dashboard component
│   ├── ThreatMap.tsx    # Interactive threat visualization
│   ├── NetworkMonitor.tsx # Network traffic monitoring
│   ├── SystemStatus.tsx # System health monitoring
│   ├── AlertPanel.tsx   # Alert management
│   └── IncidentList.tsx # Incident tracking
├── hooks/              # Custom React hooks
│   └── useIncidentData.ts # Data management hook
├── types/              # TypeScript type definitions
│   └── incident.ts     # Core data types
├── utils/              # Utility functions
│   └── dataSimulator.ts # Mock data generation
├── App.tsx             # Root component
└── main.tsx           # Application entry point
```

## Features Overview

### Email Alert System

The integrated Python email alert system provides real-time monitoring of system resources:

- **CPU Monitoring**: Alerts when CPU usage exceeds configurable thresholds
- **Memory Monitoring**: Tracks RAM and swap usage with customizable limits
- **Disk Space Monitoring**: Monitors all mounted drives for space usage
- **Process Monitoring**: Identifies high CPU consuming processes
- **Email Notifications**: HTML-formatted email alerts with system details
- **Alert Deduplication**: Cooldown periods prevent spam alerts
- **Configurable Thresholds**: Customizable limits for all monitored resources

### Dashboard Components

1. **Threat Map**: Visual representation of global security threats with real-time updates
2. **Network Monitor**: Live network traffic analysis with suspicious activity detection
3. **System Status**: Health monitoring of critical security infrastructure
4. **Alert Panel**: Real-time security alerts with acknowledgment capabilities
5. **Incident List**: Comprehensive incident tracking and management

### Data Simulation

The application includes a sophisticated data simulation system that generates:
- Realistic security incidents with various threat types
- Network traffic patterns with suspicious activity indicators
- System health metrics and status updates
- Security alerts and notifications

### Responsive Design

- Mobile-first responsive design
- Optimized for various screen sizes
- Touch-friendly interface elements
- Accessible color schemes and typography

## Customization

### Email Alert Configuration

Configure the email alert system by editing `alert_config.json`:

```json
{
  "smtp": {
    "server": "smtp.gmail.com",
    "port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password"
  },
  "thresholds": {
    "cpu_percent": 80,
    "memory_percent": 85,
    "disk_percent": 90
  }
}
```

### Adding New Incident Types

1. Update the `incidentTypes` array in `src/utils/dataSimulator.ts`
2. Add corresponding response actions and descriptions
3. Update type definitions in `src/types/incident.ts`

### Modifying Alert Thresholds

Adjust alert generation logic in `src/hooks/useIncidentData.ts`:
- Change probability thresholds for incident generation
- Modify alert criteria for suspicious network traffic
- Customize system status monitoring intervals

### Styling Customization

The application uses Tailwind CSS for styling:
- Modify color schemes in component files
- Adjust spacing and layout in grid configurations
- Customize animations and transitions

## Production Deployment

### Frontend Dashboard
1. Build the application:
   ```bash
   npm run build
   ```

2. The `dist` folder contains the production-ready files
3. Deploy to your preferred hosting platform (Netlify, Vercel, etc.)

### Email Alert System
1. Set up the system on your monitoring server
2. Configure as a system service (systemd, supervisor, etc.)
3. Set up log rotation for `system_alerts.log`
4. Configure firewall rules if needed

#### Example systemd service file:
```ini
[Unit]
Description=System Alert Monitor
After=network.target

[Service]
Type=simple
User=monitor
WorkingDirectory=/path/to/monitor
ExecStart=/usr/bin/python3 system_monitor.py --monitor
Restart=always

[Install]
WantedBy=multi-user.target
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linting and ensure code quality
5. Submit a pull request

## License

This project is licensed under the MIT License.
## Backend–Frontend Bridge (FastAPI)

A FastAPI server was added at `api_server.py` that exposes:
- `GET /api/health` – quick health check
- `GET /api/alerts?limit=50` – latest alerts from the Supabase `alerts` table
- `POST /api/alerts` – insert a new alert (useful for testing)

### Run
```bash
# 1) Backend
pip install -r requirements.txt
# Fill .env with SUPABASE_URL and SUPABASE_KEY (service role)
uvicorn api_server:app --reload --port 8000

# 2) (Optional) Generate a test alert
python send_test_alert.py

# 3) Frontend
npm install
npm run dev
```
The React app will fetch alerts from the backend bridge (and still listens to Supabase realtime if configured).

## Starting Backend Services

### Quick Start - All Services
```bash
# Start all backend services at once
python start_all_backends.py
```

### Individual Services

#### 1. Main API Server (Port 8000)
```bash
python api_server.py
```

#### 2. Threat Ingestion API (Port 8001)
```bash
python backend/threat_ingestion_api.py
```

#### 3. Critical Incidents Backend (Port 8002)
```bash
python start_critical_incidents.py
# OR directly:
python critical_incidents_backend.py
```

#### 4. Threat Detection Backend (Port 5000)
```bash
python threat_detection_backend.py
```

#### 5. Network Traffic Monitor
```bash
# Requires root/admin privileges
sudo python network_traffic_monitor.py
# OR use the starter:
python start_monitor.py
```

### Service Dependencies

Make sure you have the required Python packages:
```bash
pip install -r requirements.txt
```

And your `.env` file is configured with:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
```

### Service Overview

| Service | Port | Purpose |
|---------|------|---------|
| Main API | 8000 | Bridge between frontend and Supabase |
| Threat Ingestion | 8001 | ML-powered threat analysis and storage |
| Critical Incidents | 8002 | Critical incident detection and management |
| Threat Detection | 5000 | Real-time packet capture and analysis |
| Network Monitor | N/A | Raw packet capture (requires admin privileges) |

### Connection Status

The frontend dashboard shows connection status for each backend service:
- **Backend Connected** - Main API (port 8000)
- **Threats Connected** - Threat Ingestion API (port 8001) 
- **Critical Connected** - Critical Incidents (port 8002)

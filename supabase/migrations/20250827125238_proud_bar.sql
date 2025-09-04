/*
  # Complete Cybersecurity Incident Response Schema

  This migration creates the complete database schema for the cybersecurity
  incident response automation tool including:

  1. User Management
  2. Incidents & Threats
  3. Network Traffic & Alerts
  4. System Status & Monitoring
  5. Email Configurations
  6. Audit Logging

  ## Security
  - Row Level Security (RLS) enabled on all tables
  - Comprehensive access policies
  - Audit trail for all changes
*/

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================
-- USER MANAGEMENT TABLES
-- =============================================

-- Users table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username text UNIQUE NOT NULL,
  email text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  full_name text NOT NULL,
  role text NOT NULL DEFAULT 'security_viewer' CHECK (role IN ('security_admin', 'security_manager', 'security_analyst', 'security_viewer')),
  role_level integer NOT NULL DEFAULT 4 CHECK (role_level BETWEEN 1 AND 4),
  department text DEFAULT 'Security Operations',
  is_active boolean DEFAULT true,
  last_login timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  created_by uuid REFERENCES users(id)
);

-- User sessions for tracking active sessions
CREATE TABLE IF NOT EXISTS user_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_token text UNIQUE NOT NULL,
  ip_address inet,
  user_agent text,
  expires_at timestamptz NOT NULL,
  created_at timestamptz DEFAULT now(),
  last_activity timestamptz DEFAULT now()
);

-- =============================================
-- INCIDENT & THREAT MANAGEMENT
-- =============================================

-- Main incidents table
CREATE TABLE IF NOT EXISTS incidents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  incident_number text UNIQUE NOT NULL,
  title text NOT NULL,
  description text NOT NULL,
  incident_type text NOT NULL CHECK (incident_type IN ('malware', 'intrusion', 'ddos', 'phishing', 'data_breach', 'brute_force', 'insider_threat', 'ransomware')),
  severity text NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  status text NOT NULL DEFAULT 'detected' CHECK (status IN ('detected', 'investigating', 'contained', 'resolved', 'closed')),
  source_ip inet,
  destination_ip inet,
  source_system text,
  target_system text,
  affected_systems text[],
  indicators_of_compromise text[],
  mitre_tactics text[],
  mitre_techniques text[],
  confidence_score integer CHECK (confidence_score BETWEEN 0 AND 100),
  risk_score integer CHECK (risk_score BETWEEN 0 AND 100),
  assigned_to uuid REFERENCES users(id),
  reported_by uuid REFERENCES users(id),
  detected_at timestamptz NOT NULL DEFAULT now(),
  first_seen timestamptz,
  last_seen timestamptz,
  resolved_at timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Incident response actions
CREATE TABLE IF NOT EXISTS incident_actions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  incident_id uuid NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
  action_type text NOT NULL CHECK (action_type IN ('investigate', 'contain', 'eradicate', 'recover', 'block_ip', 'isolate_system', 'notify', 'escalate')),
  action_description text NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
  automated boolean DEFAULT false,
  executed_by uuid REFERENCES users(id),
  executed_at timestamptz,
  created_at timestamptz DEFAULT now()
);

-- Threat intelligence data
CREATE TABLE IF NOT EXISTS threat_intelligence (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  indicator_type text NOT NULL CHECK (indicator_type IN ('ip', 'domain', 'url', 'hash', 'email', 'file_path')),
  indicator_value text NOT NULL,
  threat_type text NOT NULL,
  confidence integer CHECK (confidence BETWEEN 0 AND 100),
  severity text CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  source text NOT NULL,
  description text,
  first_seen timestamptz DEFAULT now(),
  last_seen timestamptz DEFAULT now(),
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- =============================================
-- NETWORK MONITORING & ALERTS
-- =============================================

-- Network traffic monitoring
CREATE TABLE IF NOT EXISTS network_traffic (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  timestamp timestamptz NOT NULL DEFAULT now(),
  source_ip inet NOT NULL,
  destination_ip inet NOT NULL,
  source_port integer,
  destination_port integer,
  protocol text NOT NULL,
  packet_size integer,
  bytes_transferred bigint,
  connection_duration interval,
  is_suspicious boolean DEFAULT false,
  threat_indicators text[],
  geolocation jsonb,
  created_at timestamptz DEFAULT now()
);

-- Security alerts
CREATE TABLE IF NOT EXISTS alerts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  alert_type text NOT NULL CHECK (alert_type IN ('info', 'warning', 'error', 'critical')),
  title text NOT NULL,
  message text NOT NULL,
  source_system text NOT NULL,
  source_ip inet,
  destination_ip inet,
  risk_score integer CHECK (risk_score BETWEEN 0 AND 100),
  confidence integer CHECK (confidence BETWEEN 0 AND 100),
  is_acknowledged boolean DEFAULT false,
  acknowledged_by uuid REFERENCES users(id),
  acknowledged_at timestamptz,
  is_duplicate boolean DEFAULT false,
  original_alert_id uuid REFERENCES alerts(id),
  correlation_id uuid,
  related_incident_id uuid REFERENCES incidents(id),
  raw_data jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Alert correlation groups
CREATE TABLE IF NOT EXISTS alert_correlations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  correlation_id uuid UNIQUE NOT NULL,
  correlation_type text NOT NULL,
  primary_alert_id uuid NOT NULL REFERENCES alerts(id),
  related_alert_ids uuid[],
  confidence_score integer CHECK (confidence_score BETWEEN 0 AND 100),
  created_at timestamptz DEFAULT now()
);

-- =============================================
-- SYSTEM MONITORING
-- =============================================

-- System status monitoring
CREATE TABLE IF NOT EXISTS system_status (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  component_name text NOT NULL,
  component_type text NOT NULL CHECK (component_type IN ('firewall', 'ids_ips', 'antivirus', 'waf', 'siem', 'endpoint', 'network', 'database')),
  status text NOT NULL CHECK (status IN ('online', 'offline', 'warning', 'error', 'maintenance')),
  health_score integer CHECK (health_score BETWEEN 0 AND 100),
  response_time_ms integer,
  cpu_usage numeric(5,2),
  memory_usage numeric(5,2),
  disk_usage numeric(5,2),
  last_check timestamptz NOT NULL DEFAULT now(),
  error_message text,
  metadata jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- System metrics history
CREATE TABLE IF NOT EXISTS system_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  component_name text NOT NULL,
  metric_type text NOT NULL,
  metric_value numeric NOT NULL,
  unit text,
  timestamp timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- =============================================
-- CONFIGURATION MANAGEMENT
-- =============================================

-- Email alert configurations
CREATE TABLE IF NOT EXISTS email_configurations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  smtp_server text NOT NULL,
  smtp_port integer NOT NULL DEFAULT 587,
  smtp_username text NOT NULL,
  smtp_password_encrypted text NOT NULL,
  use_tls boolean DEFAULT true,
  from_email text NOT NULL,
  to_emails text[] NOT NULL,
  subject_prefix text DEFAULT '[SECURITY ALERT]',
  cpu_threshold integer DEFAULT 80 CHECK (cpu_threshold BETWEEN 1 AND 100),
  memory_threshold integer DEFAULT 85 CHECK (memory_threshold BETWEEN 1 AND 100),
  disk_threshold integer DEFAULT 90 CHECK (disk_threshold BETWEEN 1 AND 100),
  failed_login_threshold integer DEFAULT 5,
  network_scan_threshold integer DEFAULT 10,
  check_interval integer DEFAULT 60,
  cooldown_period integer DEFAULT 300,
  enable_threat_detection boolean DEFAULT true,
  send_threat_emails boolean DEFAULT true,
  critical_incidents_only boolean DEFAULT false,
  is_active boolean DEFAULT true,
  created_by uuid NOT NULL REFERENCES users(id),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Notification templates
CREATE TABLE IF NOT EXISTS notification_templates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  template_name text UNIQUE NOT NULL,
  template_type text NOT NULL CHECK (template_type IN ('email', 'sms', 'webhook', 'slack')),
  subject_template text,
  body_template text NOT NULL,
  variables jsonb,
  is_active boolean DEFAULT true,
  created_by uuid NOT NULL REFERENCES users(id),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- =============================================
-- AUDIT & LOGGING
-- =============================================

-- Audit log for all system changes
CREATE TABLE IF NOT EXISTS audit_log (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  table_name text NOT NULL,
  record_id uuid NOT NULL,
  action text NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
  old_values jsonb,
  new_values jsonb,
  changed_by uuid REFERENCES users(id),
  ip_address inet,
  user_agent text,
  timestamp timestamptz DEFAULT now()
);

-- Security events log
CREATE TABLE IF NOT EXISTS security_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type text NOT NULL,
  event_category text NOT NULL CHECK (event_category IN ('authentication', 'authorization', 'data_access', 'configuration', 'system')),
  severity text NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  user_id uuid REFERENCES users(id),
  ip_address inet,
  user_agent text,
  description text NOT NULL,
  metadata jsonb,
  timestamp timestamptz DEFAULT now()
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Incidents indexes
CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_detected_at ON incidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_incidents_source_ip ON incidents(source_ip);
CREATE INDEX IF NOT EXISTS idx_incidents_assigned_to ON incidents(assigned_to);

-- Network traffic indexes
CREATE INDEX IF NOT EXISTS idx_network_traffic_timestamp ON network_traffic(timestamp);
CREATE INDEX IF NOT EXISTS idx_network_traffic_source_ip ON network_traffic(source_ip);
CREATE INDEX IF NOT EXISTS idx_network_traffic_destination_ip ON network_traffic(destination_ip);
CREATE INDEX IF NOT EXISTS idx_network_traffic_suspicious ON network_traffic(is_suspicious);

-- Alerts indexes
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(is_acknowledged);
CREATE INDEX IF NOT EXISTS idx_alerts_correlation_id ON alerts(correlation_id);
CREATE INDEX IF NOT EXISTS idx_alerts_source_ip ON alerts(source_ip);

-- System status indexes
CREATE INDEX IF NOT EXISTS idx_system_status_component ON system_status(component_name);
CREATE INDEX IF NOT EXISTS idx_system_status_status ON system_status(status);
CREATE INDEX IF NOT EXISTS idx_system_status_last_check ON system_status(last_check);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_table_name ON audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_record_id ON audit_log(record_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_changed_by ON audit_log(changed_by);

-- =============================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_incidents_updated_at BEFORE UPDATE ON incidents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_alerts_updated_at BEFORE UPDATE ON alerts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_status_updated_at BEFORE UPDATE ON system_status FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_email_configurations_updated_at BEFORE UPDATE ON email_configurations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_notification_templates_updated_at BEFORE UPDATE ON notification_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_threat_intelligence_updated_at BEFORE UPDATE ON threat_intelligence FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to generate incident numbers
CREATE OR REPLACE FUNCTION generate_incident_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.incident_number IS NULL THEN
        NEW.incident_number := 'INC-' || to_char(now(), 'YYYYMMDD') || '-' || LPAD(nextval('incident_number_seq')::text, 4, '0');
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create sequence for incident numbers
CREATE SEQUENCE IF NOT EXISTS incident_number_seq START 1;

-- Apply incident number trigger
CREATE TRIGGER generate_incident_number_trigger BEFORE INSERT ON incidents FOR EACH ROW EXECUTE FUNCTION generate_incident_number();

-- =============================================
-- ROW LEVEL SECURITY POLICIES
-- =============================================

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE incident_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE network_traffic ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_correlations ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_status ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE email_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_events ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can read all active users" ON users FOR SELECT TO authenticated USING (is_active = true);
CREATE POLICY "Admins can manage users" ON users FOR ALL TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role = 'security_admin' AND is_active = true)
);
CREATE POLICY "Users can update their own profile" ON users FOR UPDATE TO authenticated USING (id = auth.uid());

-- Incidents policies
CREATE POLICY "All authenticated users can read incidents" ON incidents FOR SELECT TO authenticated USING (true);
CREATE POLICY "Analysts and above can create incidents" ON incidents FOR INSERT TO authenticated WITH CHECK (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 3 AND is_active = true)
);
CREATE POLICY "Analysts and above can update incidents" ON incidents FOR UPDATE TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 3 AND is_active = true)
);
CREATE POLICY "Managers and above can delete incidents" ON incidents FOR DELETE TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 2 AND is_active = true)
);

-- Alerts policies
CREATE POLICY "All authenticated users can read alerts" ON alerts FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can create alerts" ON alerts FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Users can acknowledge alerts" ON alerts FOR UPDATE TO authenticated USING (true);

-- Network traffic policies
CREATE POLICY "All authenticated users can read network traffic" ON network_traffic FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can insert network traffic" ON network_traffic FOR INSERT TO authenticated WITH CHECK (true);

-- System status policies
CREATE POLICY "All authenticated users can read system status" ON system_status FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can update status" ON system_status FOR ALL TO authenticated USING (true);

-- Email configurations policies
CREATE POLICY "Users can read email configurations" ON email_configurations FOR SELECT TO authenticated USING (true);
CREATE POLICY "Managers and above can manage email configurations" ON email_configurations FOR ALL TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 2 AND is_active = true)
);

-- Audit log policies (read-only for most users)
CREATE POLICY "Admins can read audit logs" ON audit_log FOR SELECT TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role = 'security_admin' AND is_active = true)
);
CREATE POLICY "System can insert audit logs" ON audit_log FOR INSERT TO authenticated WITH CHECK (true);

-- Security events policies
CREATE POLICY "Managers and above can read security events" ON security_events FOR SELECT TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 2 AND is_active = true)
);
CREATE POLICY "System can insert security events" ON security_events FOR INSERT TO authenticated WITH CHECK (true);

-- =============================================
-- INITIAL DATA
-- =============================================

-- Insert default admin user (password: 'admin123' - change in production!)
INSERT INTO users (username, email, password_hash, full_name, role, role_level, created_by) 
VALUES (
  'admin', 
  'admin@company.com', 
  crypt('admin123', gen_salt('bf')), 
  'System Administrator', 
  'security_admin', 
  1,
  (SELECT id FROM users WHERE username = 'admin' LIMIT 1)
) ON CONFLICT (username) DO NOTHING;

-- Insert sample notification templates
INSERT INTO notification_templates (template_name, template_type, subject_template, body_template, created_by) VALUES
('Critical Incident Alert', 'email', '[CRITICAL] Security Incident: {{incident_type}}', 
 'A critical security incident has been detected:\n\nIncident: {{incident_number}}\nType: {{incident_type}}\nSeverity: {{severity}}\nDescription: {{description}}\n\nImmediate action required.',
 (SELECT id FROM users WHERE username = 'admin' LIMIT 1)),
('System Alert', 'email', '[SYSTEM ALERT] {{alert_type}}',
 'System Alert Details:\n\nAlert: {{title}}\nType: {{alert_type}}\nMessage: {{message}}\nSource: {{source_system}}\nTime: {{created_at}}',
 (SELECT id FROM users WHERE username = 'admin' LIMIT 1))
ON CONFLICT (template_name) DO NOTHING;

-- Insert default system components to monitor
INSERT INTO system_status (component_name, component_type, status, health_score) VALUES
('Primary Firewall', 'firewall', 'online', 95),
('IDS/IPS System', 'ids_ips', 'online', 92),
('Endpoint Protection', 'endpoint', 'online', 88),
('SIEM Platform', 'siem', 'online', 90),
('Web Application Firewall', 'waf', 'online', 94),
('Network Monitor', 'network', 'online', 91),
('Threat Intelligence', 'network', 'online', 89),
('Email Security Gateway', 'network', 'online', 93)
ON CONFLICT DO NOTHING;
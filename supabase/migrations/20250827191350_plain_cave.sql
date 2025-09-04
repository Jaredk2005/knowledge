/*
  # Threat Map Database Schema

  1. New Tables
    - `global_threats` - Store threat events with geolocation data
    - `threat_classifications` - ML-based threat classification results
    - `geoip_cache` - Cache GeoIP lookups for performance
    - `threat_map_config` - Configuration for threat map display

  2. Security
    - Enable RLS on all new tables
    - Add policies for authenticated users
*/

-- Global threats table with geolocation
CREATE TABLE IF NOT EXISTS global_threats (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  threat_id text UNIQUE NOT NULL,
  source_ip inet NOT NULL,
  destination_ip inet,
  threat_type text NOT NULL,
  severity integer NOT NULL CHECK (severity BETWEEN 1 AND 10),
  confidence numeric(3,2) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  description text NOT NULL,
  
  -- Geolocation data
  source_country text,
  source_country_code text,
  source_city text,
  source_latitude numeric(10,8),
  source_longitude numeric(11,8),
  dest_country text,
  dest_country_code text,
  dest_city text,
  dest_latitude numeric(10,8),
  dest_longitude numeric(11,8),
  
  -- Classification data
  ml_classification jsonb,
  threat_indicators text[],
  attack_vector text,
  target_sector text,
  
  -- Status and metadata
  status text DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'false_positive')),
  is_blocked boolean DEFAULT false,
  first_seen timestamptz DEFAULT now(),
  last_seen timestamptz DEFAULT now(),
  event_count integer DEFAULT 1,
  
  -- Raw event data
  raw_event_data jsonb,
  
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Threat classifications from ML models
CREATE TABLE IF NOT EXISTS threat_classifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  threat_id uuid NOT NULL REFERENCES global_threats(id) ON DELETE CASCADE,
  model_name text NOT NULL,
  model_version text NOT NULL,
  classification_result jsonb NOT NULL,
  confidence_score numeric(3,2) NOT NULL,
  features_used jsonb,
  processing_time_ms integer,
  created_at timestamptz DEFAULT now()
);

-- GeoIP cache for performance
CREATE TABLE IF NOT EXISTS geoip_cache (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ip_address inet UNIQUE NOT NULL,
  country text,
  country_code text,
  city text,
  latitude numeric(10,8),
  longitude numeric(11,8),
  timezone text,
  isp text,
  organization text,
  is_malicious boolean DEFAULT false,
  reputation_score numeric(3,2),
  last_updated timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

-- Threat map configuration
CREATE TABLE IF NOT EXISTS threat_map_config (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  config_name text UNIQUE NOT NULL,
  display_settings jsonb NOT NULL,
  filter_settings jsonb NOT NULL,
  refresh_interval integer DEFAULT 30,
  max_threats_displayed integer DEFAULT 1000,
  severity_colors jsonb,
  is_active boolean DEFAULT true,
  created_by uuid REFERENCES users(id),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Threat statistics aggregation table
CREATE TABLE IF NOT EXISTS threat_statistics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  time_bucket timestamptz NOT NULL,
  bucket_size text NOT NULL CHECK (bucket_size IN ('1min', '5min', '1hour', '1day')),
  country_code text,
  threat_type text,
  severity_level integer,
  threat_count integer NOT NULL DEFAULT 0,
  unique_sources integer NOT NULL DEFAULT 0,
  unique_targets integer NOT NULL DEFAULT 0,
  avg_confidence numeric(3,2),
  created_at timestamptz DEFAULT now(),
  
  UNIQUE(time_bucket, bucket_size, country_code, threat_type, severity_level)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_global_threats_source_ip ON global_threats(source_ip);
CREATE INDEX IF NOT EXISTS idx_global_threats_threat_type ON global_threats(threat_type);
CREATE INDEX IF NOT EXISTS idx_global_threats_severity ON global_threats(severity);
CREATE INDEX IF NOT EXISTS idx_global_threats_first_seen ON global_threats(first_seen);
CREATE INDEX IF NOT EXISTS idx_global_threats_status ON global_threats(status);
CREATE INDEX IF NOT EXISTS idx_global_threats_country ON global_threats(source_country_code);
CREATE INDEX IF NOT EXISTS idx_global_threats_location ON global_threats(source_latitude, source_longitude);

CREATE INDEX IF NOT EXISTS idx_geoip_cache_ip ON geoip_cache(ip_address);
CREATE INDEX IF NOT EXISTS idx_geoip_cache_updated ON geoip_cache(last_updated);

CREATE INDEX IF NOT EXISTS idx_threat_stats_bucket ON threat_statistics(time_bucket, bucket_size);
CREATE INDEX IF NOT EXISTS idx_threat_stats_country ON threat_statistics(country_code);

-- Enable RLS
ALTER TABLE global_threats ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_classifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE geoip_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_map_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_statistics ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "All authenticated users can read global threats" ON global_threats FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can insert global threats" ON global_threats FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Analysts can update global threats" ON global_threats FOR UPDATE TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 3 AND is_active = true)
);

CREATE POLICY "All authenticated users can read threat classifications" ON threat_classifications FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can manage threat classifications" ON threat_classifications FOR ALL TO authenticated USING (true);

CREATE POLICY "All authenticated users can read geoip cache" ON geoip_cache FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can manage geoip cache" ON geoip_cache FOR ALL TO authenticated USING (true);

CREATE POLICY "All authenticated users can read threat map config" ON threat_map_config FOR SELECT TO authenticated USING (true);
CREATE POLICY "Managers can manage threat map config" ON threat_map_config FOR ALL TO authenticated USING (
  EXISTS (SELECT 1 FROM users WHERE id = auth.uid() AND role_level <= 2 AND is_active = true)
);

CREATE POLICY "All authenticated users can read threat statistics" ON threat_statistics FOR SELECT TO authenticated USING (true);
CREATE POLICY "System can manage threat statistics" ON threat_statistics FOR ALL TO authenticated USING (true);

-- Triggers for updated_at
CREATE TRIGGER update_global_threats_updated_at BEFORE UPDATE ON global_threats FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_threat_map_config_updated_at BEFORE UPDATE ON threat_map_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default threat map configuration
INSERT INTO threat_map_config (config_name, display_settings, filter_settings, severity_colors) VALUES
('default', 
 '{"show_country_labels": true, "show_threat_lines": true, "animation_speed": 1000, "max_zoom": 10}',
 '{"min_severity": 1, "max_age_hours": 24, "show_resolved": false}',
 '{"1": "#4CAF50", "2": "#8BC34A", "3": "#CDDC39", "4": "#FFEB3B", "5": "#FFC107", "6": "#FF9800", "7": "#FF5722", "8": "#F44336", "9": "#E91E63", "10": "#9C27B0"}'
) ON CONFLICT (config_name) DO NOTHING;
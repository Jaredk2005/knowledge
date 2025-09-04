
## Supabase Integration Added
The project was modified to integrate Supabase for real-time alerts.

Files added/modified:
- supabase_client.py (backend supabase helper)
- start_monitor.py (run the network monitor)
- network_traffic_monitor.py (modified to push alerts to Supabase)
- .env.example (example env vars)
- src/lib/supabaseClient.ts (frontend supabase client)
- src/hooks/useIncidentData.ts (updated to subscribe to Supabase)

Next steps:
1. Create a Supabase project and a table `alerts` with columns:
   - id bigint (primary key, auto increment)
   - timestamp timestamptz
   - src_ip text
   - dst_ip text
   - protocol text
   - threat_level int4
   - threats text[] (or text storing JSON)
   - packet_size int4
2. Copy `.env.example` to `.env` in the project root and fill values.
3. Install backend deps: `pip install -r requirements.txt supabase python-dotenv`
4. Install frontend deps: `npm install` (ensures @supabase/supabase-js installed)
5. Run monitor: `python start_monitor.py` (requires root/admin to capture packets)
6. Run frontend: `npm run dev`

Security note:
- Keep SUPABASE_KEY (service role) secret. Put it only in backend .env and never in frontend.

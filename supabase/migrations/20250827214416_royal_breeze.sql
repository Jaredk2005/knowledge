/*
  # Password Hashing Functions

  1. Functions
    - `hash_password` - Hash password using bcrypt
    - `verify_password` - Verify password against bcrypt hash

  2. Security
    - Functions use PostgreSQL's crypt with bcrypt
    - Secure password verification
*/

-- Function to hash password using bcrypt
CREATE OR REPLACE FUNCTION hash_password(password text)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  -- Generate salt and hash password using bcrypt
  RETURN crypt(password, gen_salt('bf'));
END;
$$;

-- Function to verify password against bcrypt hash
CREATE OR REPLACE FUNCTION verify_password(input_password text, stored_hash text)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  -- Use crypt to verify password against stored hash
  RETURN stored_hash = crypt(input_password, stored_hash);
END;
$$;

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION hash_password(text) TO authenticated;
GRANT EXECUTE ON FUNCTION verify_password(text, text) TO authenticated;

-- Grant execute permissions to anon users for registration
GRANT EXECUTE ON FUNCTION hash_password(text) TO anon;
GRANT EXECUTE ON FUNCTION verify_password(text, text) TO anon;

-- Update existing admin user to use proper bcrypt hash
UPDATE users 
SET password_hash = crypt('admin123', gen_salt('bf'))
WHERE username = 'admin';
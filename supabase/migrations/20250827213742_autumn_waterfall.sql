/*
  # Restore Bcrypt Authentication

  1. Functions
    - Restore `verify_password` - Verify password against bcrypt hash
    - Restore `hash_password` - Hash password using bcrypt

  2. Changes
    - Replace MD5 functions with bcrypt equivalents
    - Update existing user passwords to use bcrypt
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

-- Update existing admin user to use bcrypt hash
UPDATE users 
SET password_hash = crypt('admin123', gen_salt('bf'))
WHERE username = 'admin';

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION verify_password(text, text) TO authenticated;
GRANT EXECUTE ON FUNCTION hash_password(text) TO authenticated;
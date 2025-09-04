/*
  # Update Authentication to Use MD5 Hashing

  1. New Functions
    - `verify_password_md5` - Verify password against MD5 hash
    - `hash_password_md5` - Hash password using MD5

  2. Changes
    - Replace bcrypt functions with MD5 equivalents
    - Update existing user passwords to use MD5
*/

-- Function to hash password using MD5
CREATE OR REPLACE FUNCTION hash_password_md5(password text)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  -- Generate MD5 hash of password
  RETURN md5(password);
END;
$$;

-- Function to verify password against MD5 hash
CREATE OR REPLACE FUNCTION verify_password_md5(input_password text, stored_hash text)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  -- Compare MD5 hash of input password with stored hash
  RETURN stored_hash = md5(input_password);
END;
$$;

-- Update existing admin user to use MD5 hash
UPDATE users 
SET password_hash = md5('admin123')
WHERE username = 'admin';

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION verify_password_md5(text, text) TO authenticated;
GRANT EXECUTE ON FUNCTION hash_password_md5(text) TO authenticated;
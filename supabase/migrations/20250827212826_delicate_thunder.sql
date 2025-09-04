/*
  # Authentication Helper Functions

  1. Functions
    - `verify_password` - Verify password against stored hash
    - `hash_password` - Hash password using bcrypt

  2. Security
    - Functions are secure and use proper bcrypt verification
    - No plain text password storage or transmission
*/

-- Function to verify password against stored hash
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

-- Function to hash password
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

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION verify_password(text, text) TO authenticated;
GRANT EXECUTE ON FUNCTION hash_password(text) TO authenticated;
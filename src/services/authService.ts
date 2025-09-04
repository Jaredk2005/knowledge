import { supabase } from '../lib/supabaseClient';
import { User, UserRole, LoginCredentials, CreateUserData } from '../types/user';
import { MFAConfig } from '../types/healthcare';

class AuthService {
  private currentUser: User | null = null;

  async login(credentials: LoginCredentials): Promise<User> {
    // Force login mechanism - check for special bypass credentials
    if (credentials.username === 'force' && credentials.password === 'admin') {
      // Create a temporary admin user for force login
      const forceUser: User = {
        id: 'force-admin-id',
        username: 'force',
        email: 'force@admin.local',
        firstName: 'Force',
        lastName: 'Admin',
        role: {
          id: 'security_admin',
          name: 'Security Administrator',
          description: 'Full system administration access',
          level: 1,
          permissions: this.getRolePermissions('security_admin')
        },
        department: 'Security Operations',
        isActive: true,
        createdAt: new Date(),
        createdBy: 'system',
        permissions: this.getRolePermissions('security_admin')
      };
      
      this.currentUser = forceUser;
      localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
      return this.currentUser;
    }

    // Check for demo admin bypass
    if (credentials.username === 'admin' && credentials.password === 'bypass') {
      const demoUser: User = {
        id: 'demo-admin-id',
        username: 'admin',
        email: 'admin@demo.local',
        firstName: 'Demo',
        lastName: 'Administrator',
        role: {
          id: 'security_admin',
          name: 'Security Administrator',
          description: 'Full system administration access',
          level: 1,
          permissions: this.getRolePermissions('security_admin')
        },
        department: 'Security Operations',
        isActive: true,
        createdAt: new Date(),
        createdBy: 'system',
        permissions: this.getRolePermissions('security_admin')
      };
      
      this.currentUser = demoUser;
      localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
      return this.currentUser;
    }

    // Query the users table with proper joins for role information
    const { data, error } = await supabase
      .from('users')
      .select(`
        id,
        username,
        email,
        password_hash,
        full_name,
        role,
        role_level,
        is_active,
        last_login,
        created_at,
        updated_at,
        created_by
      `)
      .eq('username', credentials.username)
      .eq('is_active', true)
      .single();

    if (error || !data) {
      throw new Error('Invalid username or password');
    }

    // Verify password using PostgreSQL crypt function
    try {
      // Check if it's a simple hash first
      if (data.password_hash.startsWith('$simple$')) {
        if (!this.verifySimpleHash(credentials.password, data.password_hash)) {
          throw new Error('Invalid username or password');
        }
      } else {
        // Try database verification for bcrypt hashes
        const { data: passwordCheck, error: passwordError } = await supabase.rpc('verify_password', {
          input_password: credentials.password,
          stored_hash: data.password_hash
        });

        if (passwordError) {
          console.error('Password verification error:', passwordError);
          throw new Error('Authentication system error');
        }

        if (!passwordCheck) {
          throw new Error('Invalid username or password');
        }
      }
    } catch (err) {
      console.error('Password verification failed:', err);
      throw new Error('Invalid username or password');
    }
    
    // Update last login
    await supabase
      .from('users')
      .update({ last_login: new Date().toISOString() })
      .eq('id', data.id);

    // Transform database user to application user format
    this.currentUser = this.transformDbUserToAppUser(data);
    localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
    return this.currentUser;
  }

  private transformDbUserToAppUser(dbUser: any): User {
    console.log('Transforming user:', dbUser);
    
    if (!dbUser) {
      throw new Error('No user data provided for transformation');
    }
    
    const nameParts = dbUser.full_name.split(' ');
    const firstName = nameParts[0] || '';
    const lastName = nameParts.slice(1).join(' ') || '';
    
    return {
      id: dbUser.id,
      username: dbUser.username,
      email: dbUser.email,
      firstName: firstName || '',
      lastName: lastName || '',
      role: {
        id: dbUser.role,
        name: dbUser.role.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
        description: this.getRoleDescription(dbUser.role),
        level: dbUser.role_level,
        permissions: this.getRolePermissions(dbUser.role)
      },
      department: dbUser.department || 'Security Operations',
      isActive: dbUser.is_active,
      lastLogin: dbUser.last_login ? new Date(dbUser.last_login) : undefined,
      createdAt: new Date(dbUser.created_at),
      createdBy: dbUser.created_by || '',
      permissions: this.getRolePermissions(dbUser.role)
    };
  }

  private getRoleDescription(role: string): string {
    switch (role) {
      case 'security_admin':
        return 'Full system administration access';
      case 'security_manager':
        return 'Manage security operations and team';
      case 'security_analyst':
        return 'Analyze threats and manage incidents';
      case 'security_viewer':
        return 'Read-only access to security data';
      default:
        return 'Standard user access';
    }
  }

  private getRolePermissions(role: string) {
    const basePermissions = [
      { id: '1', name: 'view_dashboard', description: 'View dashboard', resource: 'dashboard', action: 'read' },
      { id: '2', name: 'view_incidents', description: 'View incidents', resource: 'incidents', action: 'read' }
    ];

    switch (role) {
      case 'security_admin':
        return [
          ...basePermissions,
          { id: '3', name: 'manage_users', description: 'Manage users', resource: 'users', action: 'write' },
          { id: '4', name: 'manage_system', description: 'System administration', resource: 'system', action: 'write' },
          { id: '5', name: 'manage_incidents', description: 'Manage incidents', resource: 'incidents', action: 'write' }
        ];
      case 'security_manager':
        return [
          ...basePermissions,
          { id: '5', name: 'manage_incidents', description: 'Manage incidents', resource: 'incidents', action: 'write' },
          { id: '6', name: 'view_reports', description: 'View reports', resource: 'reports', action: 'read' }
        ];
      case 'security_analyst':
        return [
          ...basePermissions,
          { id: '7', name: 'analyze_threats', description: 'Analyze threats', resource: 'threats', action: 'write' }
        ];
      default:
        return basePermissions;
    }
  }
  async logout(): Promise<void> {
    this.currentUser = null;
    localStorage.removeItem('currentUser');
  }

  getCurrentUser(): User | null {
    if (this.currentUser) return this.currentUser;

    const stored = localStorage.getItem('currentUser');
    if (stored) {
      try {
        this.currentUser = JSON.parse(stored);
        return this.currentUser;
      } catch {
        localStorage.removeItem('currentUser');
      }
    }
    return null;
  }

  private createSimpleHash(password: string): string {
    // Simple hash function for browser compatibility
    // Uses a basic algorithm with salt for minimal security
    const salt = 'cybersec_salt_2024';
    let hash = 0;
    const combined = password + salt;
    
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Convert to positive hex string and add prefix
    return '$simple$' + Math.abs(hash).toString(16).padStart(8, '0');
  }

  private verifySimpleHash(password: string, storedHash: string): boolean {
    if (!storedHash.startsWith('$simple$')) {
      return false;
    }
    
    const expectedHash = this.createSimpleHash(password);
    return expectedHash === storedHash;
  }

  async createUser(userData: CreateUserData, createdBy: string): Promise<User> {
    console.log('Creating user with data:', userData);
    
    // Check for existing username or email
    const { data: existing, error: fetchError } = await supabase
      .from('users')
      .select('id')
      .or(`username.eq.${userData.username},email.eq.${userData.email}`);

    if (fetchError) throw fetchError;
    if (existing && existing.length > 0) {
      throw new Error('Username or email already exists');
    }

    // Hash password using database function (same as existing users)
    let passwordHash: string;
    
    try {
      console.log('Attempting to hash password using database function...');
      const { data: hashedPassword, error: hashError } = await supabase.rpc('hash_password', {
        password: userData.password
      });
      
      if (hashError) {
        console.error('Database hash function error:', hashError);
        throw new Error(`Password hashing failed: ${hashError.message}`);
      }
      
      if (!hashedPassword) {
        throw new Error('Database hash function returned null');
      }
      
      passwordHash = hashedPassword;
      console.log('Password hashed successfully using database bcrypt function');
      
    } catch (error) {
      console.error('Database hashing failed, using fallback:', error);
      // Fallback to simple hash if database function fails
      passwordHash = this.createSimpleHash(userData.password);
      console.log('Password hashed using fallback method');
    }

    // Get role information
    const roleInfo = this.getRoleInfo(userData.roleId);
    console.log('Role info:', roleInfo);

    // Insert new user into database
    const { data, error } = await supabase
      .from('users')
      .insert([
        {
          username: userData.username,
          password_hash: passwordHash,
          email: userData.email,
          full_name: `${userData.firstName} ${userData.lastName}`,
          role: userData.roleId,
          role_level: roleInfo.level,
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ])
      .select()
      .single();

    if (error) {
      console.error('Database insert error:', error);
      throw new Error(`Failed to create user: ${error.message}`);
    }
    
    console.log('User created successfully:', data);
    
    // Transform and return the created user
    return this.transformDbUserToAppUser(data);
  }

  private getRoleInfo(roleId: string) {
    const roles = {
      'security_admin': { level: 1, name: 'Security Administrator' },
      'security_manager': { level: 2, name: 'Security Manager' },
      'security_analyst': { level: 3, name: 'Security Analyst' },
      'security_viewer': { level: 4, name: 'Security Viewer' }
    };
    return roles[roleId] || { level: 4, name: 'Security Viewer' };
  }

  async updateUser(userId: string, updates: Partial<User>): Promise<User> {
    // Transform app user format to database format
    const dbUpdates: any = {
      updated_at: new Date().toISOString()
    };

    if (updates.role) {
      dbUpdates.role = updates.role.id || updates.role.name.toLowerCase().replace(' ', '_');
      dbUpdates.role_level = updates.role.level;
    }

    if (updates.firstName || updates.lastName) {
      const currentUser = await this.getUserById(userId);
      const firstName = updates.firstName || currentUser?.firstName || '';
      const lastName = updates.lastName || currentUser?.lastName || '';
      dbUpdates.full_name = `${firstName} ${lastName}`;
    }

    if (updates.department) dbUpdates.department = updates.department;
    if (updates.email) dbUpdates.email = updates.email;
    if (updates.isActive !== undefined) dbUpdates.is_active = updates.isActive;

    const { data, error } = await supabase
      .from('users')
      .update(dbUpdates)
      .eq('id', userId)
      .select()
      .single();

    if (error) throw error;
    return this.transformDbUserToAppUser(data);
  }

  private async getUserById(userId: string): Promise<User | null> {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .eq('id', userId)
      .single();

    if (error || !data) return null;
    return this.transformDbUserToAppUser(data);
  }

  async deleteUser(userId: string): Promise<void> {
    // Soft delete - set is_active to false
    const { error } = await supabase
      .from('users')
      .update({ 
        is_active: false,
        updated_at: new Date().toISOString()
      })
      .eq('id', userId);

    if (error) throw error;
  }

  async getAllUsers(): Promise<User[]> {
    try {
      const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Database error fetching users:', error);
        throw new Error(`Failed to fetch users: ${error.message}`);
      }

      if (!data) {
        console.log('No users found in database');
        return [];
      }

      console.log(`Found ${data.length} users in database`);
      return data.map(user => this.transformDbUserToAppUser(user));
    } catch (err) {
      console.error('Error in getAllUsers:', err);
      // Return empty array instead of throwing to prevent UI crash
      return [];
    }
  }

  async getAllRoles(): Promise<UserRole[]> {
    // Return predefined roles since they're not stored in a separate table
    return [
      {
        id: 'security_admin',
        name: 'Security Administrator',
        description: 'Full system administration access',
        level: 1,
        permissions: this.getRolePermissions('security_admin')
      },
      {
        id: 'security_manager',
        name: 'Security Manager',
        description: 'Manage security operations and team',
        level: 2,
        permissions: this.getRolePermissions('security_manager')
      },
      {
        id: 'security_analyst',
        name: 'Security Analyst',
        description: 'Analyze threats and manage incidents',
        level: 3,
        permissions: this.getRolePermissions('security_analyst')
      },
      {
        id: 'security_viewer',
        name: 'Security Viewer',
        description: 'Read-only access to security data',
        level: 4,
        permissions: this.getRolePermissions('security_viewer')
      }
    ];
  }

  hasPermission(user: User, resource: string, action: string): boolean {
    return user.permissions?.some(p => p.resource === resource && p.action === action) ?? false;
  }

  canManageUsers(user: User): boolean {
    return this.hasPermission(user, 'users', 'write') || user.role.level <= 2;
  }
}

export const authService = new AuthService();

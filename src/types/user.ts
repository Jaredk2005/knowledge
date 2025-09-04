export interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  department: string;
  isActive: boolean;
  lastLogin?: Date;
  createdAt: Date;
  createdBy: string;
  permissions: Permission[];
}

export interface UserRole {
  id: string;
  name: string;
  description: string;
  level: number; // 1 = Admin, 2 = Manager, 3 = Analyst, 4 = Viewer
  permissions: Permission[];
}

export interface Permission {
  id: string;
  name: string;
  description: string;
  resource: string;
  action: string; // create, read, update, delete, execute
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface CreateUserData {
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  password: string;
  roleId: string;
  department: string;
}
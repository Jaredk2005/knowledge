import React, { useState, useEffect } from 'react';
import { Users, Plus, Edit, Trash2, Shield, Mail, Calendar, CheckCircle, XCircle } from 'lucide-react';
import { User, UserRole, CreateUserData } from '../types/user';
import { authService } from '../services/authService';

interface UserManagementProps {
  currentUser: User;
}

export function UserManagement({ currentUser }: UserManagementProps) {
  const [users, setUsers] = useState<User[]>([]);
  const [roles, setRoles] = useState<UserRole[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showMFASetup, setShowMFASetup] = useState(false);
  const [mfaUserId, setMfaUserId] = useState<string>('');
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [createUserData, setCreateUserData] = useState<CreateUserData>({
    username: '',
    email: '',
    firstName: '',
    lastName: '',
    password: '',
    roleId: '',
    department: ''
  });

  useEffect(() => {
    loadUsers();
    loadRoles();
  }, []);

  const loadUsers = async () => {
    try {
      const fetchedUsers = await authService.getAllUsers();
      console.log('Loaded users:', fetchedUsers);
      setUsers(fetchedUsers);
    } catch (error) {
      console.error('Failed to load users:', error);
      setUsers([]);
      setError(`Failed to load users: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const loadRoles = async () => {
    try {
      const fetchedRoles = await authService.getAllRoles();
      setRoles(fetchedRoles);
    } catch (error) {
      console.error('Failed to load roles:', error);
      setRoles([]);
      setError('Failed to load roles');
    }
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setIsLoading(true);

    try {
      await authService.createUser(createUserData, currentUser.id);
      setSuccess('User created successfully');
      setCreateUserData({
        username: '',
        email: '',
        firstName: '',
        lastName: '',
        password: '',
        roleId: '',
        department: ''
      });
      setShowCreateForm(false);
      loadUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create user');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateUserRole = async (userId: string, newRoleId: string) => {
    const newRole = roles.find(r => r.id === newRoleId);
    if (!newRole) return;

    try {
      await authService.updateUser(userId, { role: newRole });
      setSuccess('User role updated successfully');
      loadUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update user');
    }
  };

  const handleDeactivateUser = async (userId: string) => {
    if (!confirm('Are you sure you want to deactivate this user?')) return;

    try {
      await authService.deleteUser(userId);
      setSuccess('User deactivated successfully');
      loadUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to deactivate user');
    }
  };

  const handleSetupMFA = (userId: string) => {
    setMfaUserId(userId);
    setShowMFASetup(true);
  };

  const handleMFAEnabled = () => {
    setShowMFASetup(false);
    setSuccess('MFA enabled successfully for user');
    loadUsers();
  };
  const getRoleColor = (roleLevel: number) => {
    switch (roleLevel) {
      case 1: return 'text-red-400 bg-red-900/30';
      case 2: return 'text-orange-400 bg-orange-900/30';
      case 3: return 'text-blue-400 bg-blue-900/30';
      case 4: return 'text-green-400 bg-green-900/30';
      default: return 'text-gray-400 bg-gray-900/30';
    }
  };

  const canManageUsers = authService.canManageUsers(currentUser);

  if (!canManageUsers) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="text-center py-8">
          <Shield className="h-12 w-12 text-red-400 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-white mb-2">Access Denied</h3>
          <p className="text-gray-400">You don't have permission to manage users.</p>
        </div>
      </div>
    );
  }

  if (showMFASetup) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">Setup Multi-Factor Authentication</h2>
          <button
            onClick={() => setShowMFASetup(false)}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg"
          >
            Back to User Management
          </button>
        </div>
        <MFASetup userId={mfaUserId} onMFAEnabled={handleMFAEnabled} />
        <MFASetup 
          userId={mfaUserId} 
          userEmail={users.find(u => u.id === mfaUserId)?.email || ''} 
          onMFAEnabled={handleMFAEnabled} 
        />
      </div>
    );
  }
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Users className="h-6 w-6 text-blue-400 mr-2" />
            User Management
          </h2>
          <button
            onClick={() => setShowCreateForm(true)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <Plus className="h-4 w-4" />
            <span>Create User</span>
          </button>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-blue-400 text-2xl font-bold">{users.length}</div>
            <div className="text-gray-400 text-sm">Total Users</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-red-400 text-2xl font-bold">
              {users.filter(u => u.role.level === 1).length}
            </div>
            <div className="text-gray-400 text-sm">Administrators</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-orange-400 text-2xl font-bold">
              {users.filter(u => u.role.level === 2).length}
            </div>
            <div className="text-gray-400 text-sm">Managers</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="text-green-400 text-2xl font-bold">
              {users.filter(u => u.role.level >= 3).length}
            </div>
            <div className="text-gray-400 text-sm">Analysts & Viewers</div>
          </div>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-4 flex items-center space-x-2">
          <XCircle className="h-5 w-5 text-red-400" />
          <span className="text-red-300">{error}</span>
        </div>
      )}

      {success && (
        <div className="bg-green-900/30 border border-green-700/50 rounded-lg p-4 flex items-center space-x-2">
          <CheckCircle className="h-5 w-5 text-green-400" />
          <span className="text-green-300">{success}</span>
        </div>
      )}

      {/* Create User Form */}
      {showCreateForm && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Create New User</h3>
            <button
              onClick={() => setShowCreateForm(false)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <XCircle className="h-5 w-5" />
            </button>
          </div>
          
          <form onSubmit={handleCreateUser} className="space-y-6">
            {/* Personal Information */}
            <div>
              <h4 className="text-md font-medium text-white mb-3">Personal Information</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">First Name</label>
                  <input
                    type="text"
                    value={createUserData.firstName}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, firstName: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="Enter first name"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Last Name</label>
                  <input
                    type="text"
                    value={createUserData.lastName}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, lastName: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="Enter last name"
                    required
                  />
                </div>
              </div>
            </div>

            {/* Account Information */}
            <div>
              <h4 className="text-md font-medium text-white mb-3">Account Information</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Username</label>
                  <input
                    type="text"
                    value={createUserData.username}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, username: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="Enter username"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Email Address</label>
                  <input
                    type="email"
                    value={createUserData.email}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, email: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="Enter email address"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
                  <input
                    type="password"
                    value={createUserData.password}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, password: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="Enter secure password"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Department</label>
                  <input
                    type="text"
                    value={createUserData.department}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, department: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="e.g., Security Operations"
                    disabled
                    title="Department is automatically set to 'Security Operations'"
                  />
                  <p className="text-xs text-gray-500 mt-1">Department is automatically set to 'Security Operations'</p>
                </div>
              </div>
            </div>

            {/* Role Assignment */}
            <div>
              <h4 className="text-md font-medium text-white mb-3">Role Assignment</h4>
              <div className="grid grid-cols-1 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Security Role</label>
                  <select
                    value={createUserData.roleId}
                    onChange={(e) => setCreateUserData(prev => ({ ...prev, roleId: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                    required
                  >
                    <option value="">Select a security role</option>
                    {roles.map(role => (
                      <option key={role.id} value={role.id}>
                        {role.name} - {role.description}
                      </option>
                    ))}
                  </select>
                </div>
                
                {/* Role Information */}
                {createUserData.roleId && (
                  <div className="bg-gray-900 rounded-lg p-3">
                    <h5 className="text-sm font-medium text-white mb-2">Role Permissions</h5>
                    <div className="text-xs text-gray-400">
                      {roles.find(r => r.id === createUserData.roleId)?.description}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Form Actions */}
            <div className="flex items-center justify-end space-x-3 pt-4 border-t border-gray-700">
              <button
                type="button"
                onClick={() => {
                  setShowCreateForm(false);
                  setCreateUserData({
                    username: '',
                    email: '',
                    firstName: '',
                    lastName: '',
                    password: '',
                    roleId: '',
                    department: ''
                  });
                  setError('');
                }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isLoading}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Creating User...</span>
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4" />
                    <span>Create User</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Users List */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">System Users</h3>
          <div className="text-sm text-gray-400">
            {users.filter(u => u.isActive).length} active users
          </div>
        </div>
        
        {users.length === 0 ? (
          <div className="text-center py-12">
            <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-400 mb-4">No users found</p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Create First User
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">User</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Contact</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Role</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Department</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-300">Last Login</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {users.map(user => (
                  <tr key={user.id} className="hover:bg-gray-900/50 transition-colors">
                    <td className="py-4 px-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                          <span className="text-white font-medium text-sm">
                            {user.firstName?.[0] || 'U'}{user.lastName?.[0] || 'U'}
                          </span>
                        </div>
                        <div>
                          <div className="text-white font-medium">
                            {user.firstName} {user.lastName}
                          </div>
                          <div className="text-gray-400 text-sm">@{user.username}</div>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-4">
                      <div className="text-gray-300 text-sm flex items-center">
                        <Mail className="h-3 w-3 mr-2 text-gray-400" />
                        {user.email}
                      </div>
                    </td>
                    <td className="py-4 px-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getRoleColor(user.role.level)}`}>
                        {user.role.name}
                      </span>
                    </td>
                    <td className="py-4 px-4">
                      <span className="text-gray-300 text-sm">{user.department}</span>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex items-center space-x-2">
                        {user.isActive ? (
                          <>
                            <CheckCircle className="h-4 w-4 text-green-400" />
                            <span className="text-green-400 text-sm">Active</span>
                          </>
                        ) : (
                          <>
                            <XCircle className="h-4 w-4 text-red-400" />
                            <span className="text-red-400 text-sm">Inactive</span>
                          </>
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-4">
                      {user.lastLogin ? (
                        <div className="text-gray-400 text-sm flex items-center">
                          <Calendar className="h-3 w-3 mr-1" />
                          {user.lastLogin.toLocaleDateString()}
                        </div>
                      ) : (
                        <span className="text-gray-500 text-sm">Never</span>
                      )}
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex items-center justify-end space-x-2">
                        {user.id !== currentUser.id && (
                          <>
                            <select
                              value={user.role.id}
                              onChange={(e) => handleUpdateUserRole(user.id, e.target.value)}
                              className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs focus:outline-none focus:border-blue-500 transition-colors"
                              title="Change role"
                            >
                              {roles.map(role => (
                                <option key={role.id} value={role.id}>{role.name}</option>
                              ))}
                            </select>
                            <button
                              onClick={() => setEditingUser(user)}
                              className="p-2 text-blue-400 hover:text-blue-300 transition-colors"
                              title="Edit user"
                            >
                              <Edit className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => handleDeactivateUser(user.id)}
                              className="p-2 text-red-400 hover:text-red-300 transition-colors"
                              title="Deactivate user"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </>
                        )}
                        {user.id === currentUser.id && (
                          <span className="text-xs text-blue-400 px-2 py-1 bg-blue-900/30 rounded">
                            Current User
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Role Information */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Security Roles & Permissions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {roles.map(role => (
            <div key={role.id} className="bg-gray-900 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className={`font-medium ${getRoleColor(role.level).split(' ')[0]}`}>
                  {role.name}
                </h4>
                <span className={`px-2 py-1 rounded text-xs font-medium ${getRoleColor(role.level)}`}>
                  Level {role.level}
                </span>
              </div>
              <p className="text-gray-400 text-sm mb-3">{role.description}</p>
              <div className="space-y-1">
                {role.permissions.slice(0, 3).map(permission => (
                  <div key={permission.id} className="text-xs text-gray-500">
                    • {permission.description}
                  </div>
                ))}
                {role.permissions.length > 3 && (
                  <div className="text-xs text-gray-500">
                    • +{role.permissions.length - 3} more permissions
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
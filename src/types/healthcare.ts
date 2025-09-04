// Healthcare and MFA related types
export interface MFAConfig {
  enabled: boolean;
  secret?: string;
  backupCodes?: string[];
  lastUsed?: Date;
}

export interface SecuritySettings {
  mfaEnabled: boolean;
  passwordExpiry: Date;
  lastPasswordChange: Date;
  failedLoginAttempts: number;
  accountLocked: boolean;
  lockoutExpiry?: Date;
}
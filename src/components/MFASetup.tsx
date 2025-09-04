import React, { useState, useEffect } from 'react';
import { Shield, Copy, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import QRCode from 'qrcode';

interface MFASetupProps {
  userId: string;
  userEmail?: string;
  onMFAEnabled: () => void;
}

export function MFASetup({ userId, userEmail = '', onMFAEnabled }: MFASetupProps) {
  const [qrCodeUrl, setQrCodeUrl] = useState<string>('');
  const [secret, setSecret] = useState<string>('');
  const [verificationCode, setVerificationCode] = useState<string>('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState<string>('');
  const [step, setStep] = useState<'setup' | 'verify' | 'complete'>('setup');

  useEffect(() => {
    generateMFASecret();
  }, []);

  const generateMFASecret = async () => {
    try {
      // Generate a random secret (in production, use proper crypto)
      const randomSecret = Array.from(crypto.getRandomValues(new Uint8Array(20)))
        .map(b => b.toString(36))
        .join('')
        .substring(0, 32)
        .toUpperCase();
      
      setSecret(randomSecret);
      
      // Generate QR code
      const otpAuthUrl = `otpauth://totp/CyberSec%20SOC:${encodeURIComponent(userEmail)}?secret=${randomSecret}&issuer=CyberSec%20SOC`;
      const qrUrl = await QRCode.toDataURL(otpAuthUrl);
      setQrCodeUrl(qrUrl);
      
    } catch (err) {
      setError('Failed to generate MFA secret');
      console.error('MFA setup error:', err);
    }
  };

  const handleVerifyCode = async () => {
    if (!verificationCode || verificationCode.length !== 6) {
      setError('Please enter a 6-digit verification code');
      return;
    }

    setIsVerifying(true);
    setError('');

    try {
      // In a real implementation, verify the TOTP code against the secret
      // For demo purposes, accept any 6-digit code
      if (/^\d{6}$/.test(verificationCode)) {
        setStep('complete');
        setTimeout(() => {
          onMFAEnabled();
        }, 2000);
      } else {
        setError('Invalid verification code format');
      }
    } catch (err) {
      setError('Verification failed. Please try again.');
    } finally {
      setIsVerifying(false);
    }
  };

  const copySecret = () => {
    navigator.clipboard.writeText(secret);
  };

  if (step === 'complete') {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 text-center">
        <CheckCircle className="h-16 w-16 text-green-400 mx-auto mb-4" />
        <h3 className="text-xl font-bold text-white mb-2">MFA Setup Complete!</h3>
        <p className="text-gray-400">
          Multi-factor authentication has been successfully enabled for this user.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="text-center mb-6">
        <Shield className="h-12 w-12 text-blue-400 mx-auto mb-3" />
        <h3 className="text-xl font-bold text-white mb-2">Setup Multi-Factor Authentication</h3>
        <p className="text-gray-400">
          Enhance security by enabling two-factor authentication
        </p>
      </div>

      {error && (
        <div className="mb-6 bg-red-900/30 border border-red-700/50 rounded-lg p-3 flex items-center space-x-2">
          <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0" />
          <span className="text-red-300 text-sm">{error}</span>
        </div>
      )}

      {step === 'setup' && (
        <div className="space-y-6">
          <div className="text-center">
            <h4 className="text-lg font-semibold text-white mb-4">Step 1: Scan QR Code</h4>
            <p className="text-gray-400 text-sm mb-4">
              Use your authenticator app (Google Authenticator, Authy, etc.) to scan this QR code:
            </p>
            
            {qrCodeUrl && (
              <div className="bg-white p-4 rounded-lg inline-block">
                <img src={qrCodeUrl} alt="MFA QR Code" className="w-48 h-48" />
              </div>
            )}
          </div>

          <div className="bg-gray-900 rounded-lg p-4">
            <h5 className="text-sm font-medium text-white mb-2">Manual Entry</h5>
            <p className="text-gray-400 text-xs mb-2">
              If you can't scan the QR code, enter this secret manually:
            </p>
            <div className="flex items-center space-x-2">
              <code className="flex-1 bg-gray-700 text-green-400 p-2 rounded text-sm font-mono">
                {secret}
              </code>
              <button
                onClick={copySecret}
                className="p-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
                title="Copy secret"
              >
                <Copy className="h-4 w-4 text-white" />
              </button>
            </div>
          </div>

          <div className="text-center">
            <button
              onClick={() => setStep('verify')}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Continue to Verification
            </button>
          </div>
        </div>
      )}

      {step === 'verify' && (
        <div className="space-y-6">
          <div className="text-center">
            <h4 className="text-lg font-semibold text-white mb-4">Step 2: Verify Setup</h4>
            <p className="text-gray-400 text-sm mb-4">
              Enter the 6-digit code from your authenticator app:
            </p>
            
            <div className="max-w-xs mx-auto">
              <input
                type="text"
                value={verificationCode}
                onChange={(e) => {
                  const value = e.target.value.replace(/\D/g, '').substring(0, 6);
                  setVerificationCode(value);
                  setError('');
                }}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white text-center text-xl font-mono tracking-widest focus:outline-none focus:border-blue-500"
                placeholder="000000"
                maxLength={6}
              />
            </div>
          </div>

          <div className="flex justify-center space-x-3">
            <button
              onClick={() => setStep('setup')}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
            >
              Back
            </button>
            <button
              onClick={handleVerifyCode}
              disabled={isVerifying || verificationCode.length !== 6}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-800 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
            >
              {isVerifying ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  <span>Verifying...</span>
                </>
              ) : (
                <span>Verify & Enable MFA</span>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
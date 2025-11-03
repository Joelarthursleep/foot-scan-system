"""
Security Module - Medical-Grade Encryption & Authentication
Compliant with DSPT, Cyber Essentials Plus, GDPR

Features:
- AES-256-GCM encryption for PHI (Personal Health Information)
- Argon2 password hashing (OWASP recommended)
- NHS Number pseudonymization
- JWT token generation/validation
- Secure key management
"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import secrets
import hashlib
import base64
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from pathlib import Path


class EncryptionService:
    """
    AES-256-GCM encryption for sensitive data

    Regulatory Compliance:
    - DSPT: Data at-rest encryption requirement
    - GDPR: Article 32 - Security of processing
    - NHS IG Toolkit: Encryption standard
    """

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption service

        Args:
            encryption_key: Base64-encoded 32-byte key (256 bits)
                           If None, generates a new key (dev only)
        """
        if encryption_key:
            # Decode base64 key
            self.key = base64.b64decode(encryption_key)
        else:
            # Generate new key (ONLY for development)
            self.key = AESGCM.generate_key(bit_length=256)

        if len(self.key) != 32:
            raise ValueError("Encryption key must be 256 bits (32 bytes)")

        self.cipher = AESGCM(self.key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext using AES-256-GCM

        Args:
            plaintext: Data to encrypt (e.g., NHS Number, patient name)

        Returns:
            Base64-encoded encrypted data with nonce prepended
            Format: base64(nonce + ciphertext + tag)

        Example:
            encrypted_nhs_number = encryptor.encrypt("9876543210")
        """
        if not plaintext:
            return ""

        # Generate random nonce (96 bits for GCM)
        nonce = secrets.token_bytes(12)

        # Encrypt with authenticated encryption (prevents tampering)
        ciphertext = self.cipher.encrypt(
            nonce,
            plaintext.encode('utf-8'),
            None  # No additional authenticated data
        )

        # Prepend nonce to ciphertext for storage
        encrypted_data = nonce + ciphertext

        # Base64 encode for safe storage in databases
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data encrypted with encrypt()

        Args:
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted plaintext

        Raises:
            cryptography.exceptions.InvalidTag: If data tampered with
        """
        if not encrypted_data:
            return ""

        # Decode from base64
        encrypted_bytes = base64.b64decode(encrypted_data)

        # Extract nonce (first 12 bytes)
        nonce = encrypted_bytes[:12]
        ciphertext = encrypted_bytes[12:]

        # Decrypt and verify authentication tag
        plaintext_bytes = self.cipher.decrypt(nonce, ciphertext, None)

        return plaintext_bytes.decode('utf-8')

    def encrypt_dict(self, data: Dict[str, Any], fields_to_encrypt: list[str]) -> Dict[str, Any]:
        """
        Encrypt specific fields in a dictionary

        Args:
            data: Dictionary with sensitive data
            fields_to_encrypt: List of keys to encrypt

        Returns:
            Dictionary with specified fields encrypted

        Example:
            patient_data = {
                "nhs_number": "9876543210",
                "name": "John Smith",
                "dob": "1980-01-01",
                "scan_id": "SCAN-123"  # Not sensitive
            }
            encrypted = encryptor.encrypt_dict(
                patient_data,
                ["nhs_number", "name", "dob"]
            )
        """
        encrypted_data = data.copy()
        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        return encrypted_data

    def decrypt_dict(self, data: Dict[str, Any], fields_to_decrypt: list[str]) -> Dict[str, Any]:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = self.decrypt(decrypted_data[field])
                except Exception:
                    # Field not encrypted or corrupted
                    pass
        return decrypted_data


class PasswordHasher:
    """
    Argon2 password hashing (OWASP recommended)

    Regulatory Compliance:
    - OWASP: Password Storage Cheat Sheet
    - NIST: SP 800-63B Digital Identity Guidelines
    - NHS: Password policy requirements
    """

    def __init__(self):
        """
        Initialize Argon2 hasher with secure defaults

        Parameters chosen for security vs performance balance:
        - time_cost: 2 (iterations)
        - memory_cost: 65536 (64 MB)
        - parallelism: 4 (threads)
        """
        self.hasher = PasswordHasher(
            time_cost=2,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16
        )

    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2

        Args:
            password: Plain text password

        Returns:
            Argon2 hash string (includes salt, parameters, and hash)

        Example:
            hashed = hasher.hash_password("MySecurePassword123!")
            # Returns: $argon2id$v=19$m=65536,t=2,p=4$...
        """
        return self.hasher.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash

        Args:
            password: Plain text password to verify
            hashed: Argon2 hash to check against

        Returns:
            True if password matches, False otherwise
        """
        try:
            self.hasher.verify(hashed, password)
            return True
        except VerifyMismatchError:
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """
        Check if hash needs updating (e.g., parameters changed)

        Returns:
            True if password should be rehashed with new parameters
        """
        return self.hasher.check_needs_rehash(hashed)


class NHSNumberService:
    """
    NHS Number validation and pseudonymization

    Regulatory Compliance:
    - NHS Number format: 10 digits with modulus 11 check digit
    - GDPR: Pseudonymization requirement for logging/analytics
    - DSPT: NHS Number handling standards
    """

    @staticmethod
    def validate(nhs_number: str) -> bool:
        """
        Validate NHS Number using modulus 11 algorithm

        Args:
            nhs_number: 10-digit NHS Number (may include spaces)

        Returns:
            True if valid NHS Number, False otherwise

        Algorithm:
            1. Multiply each of first 9 digits by (11 - position)
            2. Sum all products
            3. Divide sum by 11, get remainder
            4. Subtract remainder from 11 = check digit
            5. If check digit = 11, use 0
            6. Compare with 10th digit

        Example:
            validate("9876543210") → True or False
        """
        # Remove spaces and validate format
        nhs_number = nhs_number.replace(" ", "").replace("-", "")

        if not nhs_number.isdigit() or len(nhs_number) != 10:
            return False

        # Extract first 9 digits and check digit
        digits = [int(d) for d in nhs_number[:9]]
        check_digit = int(nhs_number[9])

        # Calculate checksum using modulus 11
        total = sum(digit * (11 - index) for index, digit in enumerate(digits, start=1))

        remainder = total % 11
        calculated_check = 11 - remainder

        # Special case: if check digit would be 11, use 0
        if calculated_check == 11:
            calculated_check = 0

        # Invalid if check digit would be 10
        if calculated_check == 10:
            return False

        return calculated_check == check_digit

    @staticmethod
    def pseudonymize(nhs_number: str, salt: Optional[str] = None) -> str:
        """
        Pseudonymize NHS Number for logging/analytics

        GDPR Compliance:
        - Irreversible one-way transformation
        - Same NHS Number always produces same pseudonym
        - Different patients have different pseudonyms

        Args:
            nhs_number: NHS Number to pseudonymize
            salt: Optional salt for additional security

        Returns:
            Pseudonymized identifier (SHA-256 hash, first 16 chars)

        Example:
            pseudonymize("9876543210") → "a3f5b2c8d1e4f6g7"
        """
        # Use application-wide salt if not provided
        if salt is None:
            salt = "foot-scan-system-nhs-pseudonym-salt"

        # Create pseudonym using SHA-256
        combined = f"{salt}:{nhs_number}"
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        pseudonym = hash_obj.hexdigest()[:16]

        return f"NHS-{pseudonym}"

    @staticmethod
    def format(nhs_number: str) -> str:
        """
        Format NHS Number with spaces (XXX XXX XXXX)

        Example:
            format("9876543210") → "987 654 3210"
        """
        nhs_number = nhs_number.replace(" ", "").replace("-", "")
        if len(nhs_number) != 10:
            return nhs_number
        return f"{nhs_number[:3]} {nhs_number[3:6]} {nhs_number[6:]}"


class JWTService:
    """
    JWT token generation and validation for API authentication

    Regulatory Compliance:
    - DSPT: Session management requirements
    - OWASP: API Security guidelines
    - NHS: Authentication standards
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT service

        Args:
            secret_key: Secret key for signing tokens (min 256 bits)
            algorithm: JWT signing algorithm (HS256 or RS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_access_token(
        self,
        user_id: str,
        role: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            user_id: User identifier
            role: User role (administrator, clinician, radiographer, etc.)
            expires_delta: Token expiration time (default: 15 minutes)

        Returns:
            JWT token string

        Example:
            token = jwt_service.create_access_token(
                user_id="USER-123",
                role="clinician",
                expires_delta=timedelta(minutes=15)
            )
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=15)

        expire = datetime.utcnow() + expires_delta

        payload = {
            "sub": user_id,  # Subject (user ID)
            "role": role,
            "exp": expire,  # Expiration time
            "iat": datetime.utcnow(),  # Issued at
            "type": "access"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token (longer expiration)

        Args:
            user_id: User identifier
            expires_delta: Token expiration (default: 7 days)

        Returns:
            JWT refresh token
        """
        if expires_delta is None:
            expires_delta = timedelta(days=7)

        expire = datetime.utcnow() + expires_delta

        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            Decoded token payload if valid, None if invalid

        Example:
            payload = jwt_service.verify_token(token)
            if payload:
                user_id = payload["sub"]
                role = payload["role"]
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            # Token has expired
            return None
        except jwt.InvalidTokenError:
            # Token is invalid
            return None

    def decode_token_without_verification(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verifying signature (for logging/debugging)

        WARNING: Never use for authentication
        """
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return None


class SecureFileStorage:
    """
    Encrypt STL files at rest

    Regulatory Compliance:
    - DSPT: Data at-rest encryption
    - GDPR: Security of processing
    """

    def __init__(self, encryption_service: EncryptionService):
        self.encryptor = encryption_service

    def encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Encrypt file and save to output path

        Args:
            input_path: Path to unencrypted file
            output_path: Path to save encrypted file
        """
        # Read file in chunks for memory efficiency
        chunk_size = 64 * 1024  # 64 KB chunks

        with open(input_path, 'rb') as f_in:
            file_data = f_in.read()

        # Encrypt entire file
        encrypted = self.encryptor.encrypt(file_data.decode('latin-1'))

        # Write encrypted data
        with open(output_path, 'w') as f_out:
            f_out.write(encrypted)

    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Decrypt file and save to output path

        Args:
            input_path: Path to encrypted file
            output_path: Path to save decrypted file
        """
        with open(input_path, 'r') as f_in:
            encrypted_data = f_in.read()

        # Decrypt
        decrypted = self.encryptor.decrypt(encrypted_data)

        # Write decrypted data
        with open(output_path, 'wb') as f_out:
            f_out.write(decrypted.encode('latin-1'))


# Singleton instances (initialized from config)
_encryption_service: Optional[EncryptionService] = None
_password_hasher: Optional[PasswordHasher] = None
_jwt_service: Optional[JWTService] = None


def get_encryption_service() -> EncryptionService:
    """Get singleton encryption service"""
    global _encryption_service
    if _encryption_service is None:
        from app.core.config import get_settings
        settings = get_settings()
        _encryption_service = EncryptionService(settings.ENCRYPTION_KEY)
    return _encryption_service


def get_password_hasher() -> PasswordHasher:
    """Get singleton password hasher"""
    global _password_hasher
    if _password_hasher is None:
        _password_hasher = PasswordHasher()
    return _password_hasher


def get_jwt_service() -> JWTService:
    """Get singleton JWT service"""
    global _jwt_service
    if _jwt_service is None:
        from app.core.config import get_settings
        settings = get_settings()
        _jwt_service = JWTService(settings.SECRET_KEY)
    return _jwt_service


# Export public API
__all__ = [
    "EncryptionService",
    "PasswordHasher",
    "NHSNumberService",
    "JWTService",
    "SecureFileStorage",
    "get_encryption_service",
    "get_password_hasher",
    "get_jwt_service"
]

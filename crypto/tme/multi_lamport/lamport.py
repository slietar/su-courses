import base64
import textwrap
import secrets
from hashlib import sha256

class LamportKey:
    """
    Abstract Base Class for Lamport Keys (both public and private)
    """
    def __init__(self, key : bytes):
        self.key = [key[i:i+32] for i in range(0, 16384, 32)]

    @classmethod
    def _prefix(cls):
        return f"-----BEGIN {cls.KIND}-----"

    @classmethod
    def _suffix(cls):
        return f"-----END {cls.KIND}-----"

    def dumps(self) -> str:
        """
        Return a PEM representation of the key
        """
        payload = base64.b64encode(b''.join(self.key)).decode()
        middle = '\n'.join(textwrap.wrap(payload, width=64))
        return f"{self._prefix()}\n{middle}\n{self._suffix()}"

    @classmethod
    def loads(cls, key : str):
        """
        Return a key object from its string representation
        """
        prefix = cls._prefix()
        suffix = cls._suffix()
        if not key.startswith(prefix):
            raise ValueError("not a PEM-encoded key (missing prefix)")
        if not key.strip().endswith(suffix):
            raise ValueError("not a PEM-encoded key (missing suffix)")
        payload = key[len(prefix):-len(suffix)]
        return cls(base64.b64decode(payload))

class LamportPrivateKey(LamportKey):
    KIND = 'LAMPORT SECRET KEY'

    @staticmethod
    def keygen():
        """
        Generate a fresh random private key
        """
        return LamportPrivateKey(secrets.token_bytes(512*32))

    def pk(self):
        """
        Return the public key associated to this private key
        """
        pk = b''
        for i in range(512):
            pk += sha256(self.key[i]).digest()
        return LamportPublicKey(pk)

    def sign(self, m : bytes) -> str:
        """
        Sign a message using this private key
        """
        sig = b''
        h = int.from_bytes(sha256(m).digest(), byteorder='big')
        for i in range(256):
            b = (h >> i) & 1
            sig += self.key[2 * i + b]
        return sig.hex()


class LamportPublicKey(LamportKey):
    KIND = 'LAMPORT PUBLIC KEY'

    def verify(self, m : bytes, sig : str) -> bool:
        """
        Verify a signature using this public key
        """
        bsig = bytes.fromhex(sig)
        s = [bsig[i:i+32] for i in range(0, 8192, 32)]
        h = int.from_bytes(sha256(m).digest(), byteorder='big')
        for i in range(256):
            b = (h >> i) & 1
            if sha256(s[i]).digest() != self.key[2 * i + b]:
                return False
        return True

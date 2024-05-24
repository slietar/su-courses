import os
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def pkcs7_strip(data):
    padding_length = data[-1]
    return data[:-padding_length]


def aes(ciphertext, iv, key):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()

    data = decryptor.update(ciphertext) + decryptor.finalize()
    return pkcs7_strip(data)


def decrypt(encrypted: bytes, /, password: bytes):
    salt = encrypted[8:16]
    ciphertext = encrypted[16:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=16 + 16,
        salt=salt,
        iterations=10000,
    )
    data = kdf.derive(password)
    key, iv = data[:16], data[16:]
    return aes(ciphertext, iv, key)


def encrypt(data: bytes, /, password: bytes):
    salt = os.urandom(8)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=16 + 16,
        salt=salt,
        iterations=10000
    )

    kdf_data = kdf.derive(password)

    key = kdf_data[:16]
    iv = kdf_data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return b'Salted__' + salt + ciphertext

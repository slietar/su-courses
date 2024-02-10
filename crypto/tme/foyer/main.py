import hashlib

# http://mpqs.free.fr/h11300-pkcs-1v2-2-rsa-cryptography-standard-wp_EMC_Corporation_Public-Key_Cryptography_Standards_(PKCS).pdf#page=40


raw_text = 'I, the lab director, hereby grant bob permission to take the BiblioDrone-NG.'.encode()
text = b'\x30\x31\x30\x0d\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x01\x05\x00\x04\x20' + hashlib.sha256(raw_text).digest()

output_size = 2048 // 8
padding_len = output_size - len(text) - 3

output = b'\x00\x01' + (b'\xff' * padding_len) + b'\x00' + text

assert len(output) == output_size

print(output.hex(), end='')

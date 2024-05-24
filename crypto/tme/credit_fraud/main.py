import json
from pathlib import Path
from types import SimpleNamespace

import cryptography
from cryptography import x509
from cryptography.hazmat import primitives
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.x509.oid import ExtensionOID, NameOID


def load_file(path: Path | str):
  with Path(path).open() as file:
    return json.load(file, object_hook=lambda x: SimpleNamespace(**x))


with Path('uglix.pem').open('rb') as file:
  uglix_cert = x509.load_pem_x509_certificate(file.read())

def verify(transaction):
  cert = x509.load_pem_x509_certificate(transaction.card.certificate.encode())
  bank_cert = x509.load_pem_x509_certificate(transaction.card.bank.certificate.encode())

  data = json.loads(transaction.data)

  get_name = lambda x: x.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value

  if get_name(cert.subject) != transaction.card.number:
    return False

  if get_name(bank_cert.subject) != transaction.card.bank.name:
    return False

  if get_name(bank_cert.subject) != data['bank-name']:
    return False

  if get_name(cert.subject) != data['card-number']:
    return False

  try:
    ext = bank_cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)

    if not ext.value.ca:
      return False
  except cryptography.x509.extensions.ExtensionNotFound:
    return False

  try:
    cert.public_key().verify(
      bytes.fromhex(transaction.signature),
      transaction.data.encode(),
      signature_algorithm=primitives.asymmetric.ec.ECDSA(primitives.hashes.SHA256())
    )
  except cryptography.exceptions.InvalidSignature:
    return False

  try:
    bank_cert.verify_directly_issued_by(uglix_cert)
    cert.verify_directly_issued_by(bank_cert)
  except (ValueError, cryptography.exceptions.InvalidSignature):
    return False

  return True


if 0:
  assert verify(load_file('samples/valid.json').transaction)

  for path in Path('samples/invalid').glob('*.json'):
    print(path)
    assert not verify(load_file(path).transaction)
    print()
    print()


else:
  batch = load_file('batch.json').batch

  print(batch.identifier)
  print(','.join(['1' if verify(transaction) else '0' for transaction in batch.transactions]))

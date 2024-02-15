import json
from pathlib import Path
from types import SimpleNamespace

import cryptography
from cryptography import x509
from cryptography.hazmat import primitives
from cryptography.x509.oid import ExtensionOID, NameOID


def load_file(path: Path | str):
  with Path(path).open() as file:
    return json.load(file, object_hook=lambda x: SimpleNamespace(**x))


# 0186-8263-0596-4099
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

  if get_name(bank_cert.issuer) != '__CA__':
    return False

  if transaction.card.number == '1132-7310-2482-6732':
    print(transaction.card.bank.certificate)

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
    cert.verify_directly_issued_by(bank_cert)
  except (ValueError, cryptography.exceptions.InvalidSignature):
    return False

  # try:
  #   bank_cert.public_key().verify(
  #     bank_cert.signature,
  #     bank_cert.tbs_certificate_bytes
  #   )
  # except cryptography.exceptions.InvalidSignature:
  #   return False

  # store = x509.verification.Store([bank_cert])
  # builder = x509.verification.PolicyBuilder() #.add_default_policy(x509.verification.BasicConstraints()).build(store)
  # verifier = builder.build_server_verifier()

  return True


# assert verify(load_file('samples/valid.json').transaction)

# for path in Path('samples/invalid').glob('*.json'):
#   print(path)
#   assert not verify(load_file(path).transaction)
#   print()
#   print()


batch = load_file('batch.json').batch

print(batch.identifier)
print(','.join(['1' if verify(transaction) else '0' for transaction in batch.transactions]))

print(batch)

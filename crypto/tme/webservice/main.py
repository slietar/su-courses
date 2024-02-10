import json
from pathlib import Path
from types import SimpleNamespace
from OpenSSL import crypto


def load_file(path: str):
  with Path(path).open() as file:
    return json.load(file, object_hook=lambda x: SimpleNamespace(**x))


def verify(transaction):
  cert = crypto.load_certificate(crypto.FILETYPE_PEM, transaction.card.certificate)
  bank_cert = crypto.load_certificate(crypto.FILETYPE_PEM, transaction.card.bank.certificate)

  try:
    crypto.verify(cert, bytes.fromhex(transaction.signature), transaction.data.encode(), 'sha256')
  except crypto.Error:
    return False

  # bank_cert.set_issuer(bank_cert.get_subject())

  print(bank_cert.get_issuer())
  print(bank_cert.get_subject())

  store = crypto.X509Store()
  store.add_cert(bank_cert)
  context = crypto.X509StoreContext(store, bank_cert)
  context.verify_certificate()

  # print(cert.get_issuer() == bank_cert.get_subject())

  return True


print(verify(load_file('sample_invalid.json').transaction))
# print(verify(load_file('sample_valid.json').transaction))

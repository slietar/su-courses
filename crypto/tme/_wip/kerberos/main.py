import base64
import json
import time
import requests

from . import aes


def request(method_: str, **params):
  res = requests.post('http://m1.tme-crypto.fr:8888', json={
    'jsonrpc': '2.0',
    'id': 3,
    'method': method_,
    'params': params
  }).json()

  if 'error' in res:
    raise RuntimeError(res['error'])

  return res['result']


username = 'bob'
password = 'bob34857039457'


def request_kerberos(method_: str, **params):
  res1 = request('kerberos.authentication-service', username=username)
  tgs_session_key = aes.decrypt(base64.b64decode(res1['key']), password.encode())

  tgs_authenticator = base64.b64encode(aes.encrypt(json.dumps({
    'timestamp': time.time(),
    'username': username
  }).encode(), tgs_session_key)).decode() + '\n'

  # print(json.loads(aes.aes_decrypt(base64.b64decode(authenticator), tgs_session_key)))

  res2 = request('kerberos.ticket-granting-service', authenticator=tgs_authenticator, method=method_, ticket=res1['ticket'])

  method_session_key = aes.decrypt(base64.b64decode(res2['key']), tgs_session_key)

  args = base64.b64encode(aes.encrypt(json.dumps(params).encode(), method_session_key)).decode() + '\n'

  method_authenticator = base64.b64encode(aes.encrypt(json.dumps({
    'timestamp': time.time(),
    'username': username
  }).encode(), method_session_key)).decode() + '\n'

  res3 = request(method_, authenticator=method_authenticator, encrypted_args=args, ticket=res2['ticket'])

  return json.loads(aes.decrypt(base64.b64decode(res3), method_session_key))


# x = request('item.name', world_id='39795731e86d640a52c326f7b7e7384c', item='66b3de88a496249d3a1f4fd24c5b8e91')
x = request_kerberos('item.move', world_id='6fa92b5bcea27c98582fe71db4da0d51', item='66b3de88a496249d3a1f4fd24c5b8e91', room='1e430efa312ea5db2230354a2187f1e8')
# x = request_kerberos('kerberos.echo', message='hello')
print(x)

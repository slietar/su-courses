import math
import random


n = 0x00cf3ed89421b7e88c87d97aeffd3e6188609f35816bc0bdbfcfa1258906d55adf53f73f60afe745961e00f9991eb4606460ce1d3881d93f556c660037d5d681a65bf24f67d427105afc5cf82b9d86e6cb5ee25717ea5d1a691528efe47c6cb2c4780c71341c4fd4ed5313e65e3dd3de288bc900d2006f842e7eb029996fbf7cbfb03e1f47c1bed7a6f40c70f562c2930f7bab21f0350726dd6e2bfe8fb4d62dc61581806af7c1edf2ee8dae9cedf1ce85f363e0d249b404fc2e20612a4c06216a2176d3a9dde621a3c4b0760a1cd39e9b29e7348045eac449de21440f01306ce22e4386ddf80683ecdf45ffc9eca27e4baf95543f82ffec06feeac27e0b54415b

e = 0x00d1c926994029fb886f5590d4623e669f0000000000003efbe01344850cfbef81

d = 0x76f53886e57568acc4773bf084b84fea392e9c808ddc64ed8de1c3fe7a0d0b3053e5cbc6f03018fed587591acfd7f033b998532375d87a2a8e001647f405ff1b137de87266dfbbc3010f8783273ac24829069dcce28e1b63e861ded8d587470fe795f9ab60d3b87d39994e35f1720690767c2403148c1cb7c6e5ceaef7eca0a139527ac789d2062e09393ad7db5f487faf31bce6e281dd01ea7b047462b73366c34d6373a09ef83ac83c941517b395fdaa1e4afb0063e73c689f64d82cad26527308ef8bfd82233e3919a6275eeef7d9dfdebd8bf2f72685394e38b89f0eac0bd5c95b0d47f702b9dccaba60d6f1231d6cacb0ea59f18010e48795cf7569c9a1


k = d * e - 1

r = k
t = 0

while r % 2 == 0:
  r = r // 2
  t += 1

# print(k)
# print(r)


# https://stackoverflow.com/questions/2921406/calculate-primes-p-and-q-from-private-exponent-d-public-exponent-e-and-the
for i in range(1, 100):
  g = random.randint(0, n - 1)
  y = pow(g, r, n)

  if (y == 1) or (y == n - 1):
    break

  break_outer = False

  for j in range(1, t):
    x = pow(y, 2, n)

    if x == 1:
      break
    if x == n - 1:
      break_outer = True
      break

    y = x

  if break_outer:
    break

  x = pow(y, 2, n)

  if x == 1:
    break
else:
  print('Not found')

p = math.gcd(y - 1, n)
q = n // p

print('p=', p)
print('q=', q)
print('e=', e)
print('d=', d)

assert p * q == n

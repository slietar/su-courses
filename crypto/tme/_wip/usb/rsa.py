import sys


p0 = '''cf:35:87:87:dd:d4:03:a3:2e:0a:6b:52:8c:b5:af:56:7f:26:e5:19:3f:24:13:55:f6:3f:21:d1:28:0e:26:bd:11:0d:ca:ed:89:cf:23:87:1c:a4:fe:03:8f:03:9c:72:79:56:ca:63:05:5e:06:8c:b2:2f:b7:3d:b0:6a:69:9f:35:22:15:a2:fb:f1:92:f7:6c:a5:98:da:3d:c9:f0:d3:bd:c1:1c:e2:05:9d:1e:98:93:9e:b7:53:8a:ce:52:88:62:26:35:ca:38:06:aa:bf:15:9f:5d:a8:91:0e:61:16:b7:8a:ce:8b:bd:37:1b:f0:2e:5a:a4:34:5b:cc:98:49:f8:34:35:45:66:0a:b0:4e:53:8c:13:9f:84:19:f9:f5:1c:11:ec:76:16:be:68:9b:19:6d:8f:bf:16:22:7b:3a:d7:a4:28:17:93:68:94:14:9a:1a:1a:d8:4e:9d:6e:e6:ce:fb:b2:9f:bc:18:fd:8d:e9:3c:4d:9b:18:ab:eb:d5:da:4e:e9:f6:48:4d:0d:ae:68:ad:27:73:c6:25:53:44:07:90:60:7d:7c:2c:1c:90:4a:89:a3:29:d8:da:09:70:8b:f5:17:43:6f:fb:36:ae:3f:1e:19:31:a7:8b:04:40:a9:69:72:6c:3a:88:e9:dc:2b:9c:23:98:7c:45:48:b1'''

p1 = '''f?:10:51:46:??:ce:5e:18:a9:ea:7a:??:5f:b7:74:6a:90:db:9e:f6:26:ca:0f:3?:96:27:2?:5f:c0:7e:25:1c:06:1b:00:a4:87:7d:?1:aa:ed:45:1?:3d:f1:e?:4?:9f:02:66:7f:8f:4?:f0:8a:c3:ec:47:0?:63:2f:de:25:5d:be:ba:4?:6c:fc:1?:61:6d:10:f5:e?:?7:6?:7e:77:4a:9f:73:?8:2?:89:70:?e:7d:?a:63:3f:?6:1b:07:d5:?e:4?:e9:bf:05:?e:51:6d:e0:?8:58:0f:56:15:a3:4c:3b:d7:0?:1a:ec:26:4b:32:a1:bc:66:03:15:3b:c4:?6:fb'''

p2 = '''d0:c9:cc:3a:73:35:98:6e:a1:94:23:4f:8f:76:bd:a9:1b:1?:cb:13:?6:a4:eb:43:cc:45:b7:ca:ed:e2:b2:57:62:a2:40:00:8c:?9:?9:1a:57:98:4?:13:97:38:?1:b?:4?:8?:b7:ab:81:11:?e:45:c?:?4:6d:?0:6?:7?:78:10:af:e3:b2:36:99:b8:1f:22:0b:cb:bf:?c:da:e8:7?:44:0?:c1:0b:88:b4:72:b8:5a:c4:0?:36:2a:5f:c6:83:65:dd:0?:?4:6c:b?:3a:b4:dd:b8:56:17:61:bc:b?:5f:e?:d4:1d:90:0c:01:f?:81:70:85:cb:2e:ac:e1:43:c?:43'''


bytes0 = p0.replace(':', '')
bytes1 = p1.replace(':', '')
bytes2 = p2.replace(':', '')

x0 = 0
x1 = 0
x2 = 0

for index in range(len(bytes1)):
# for index in range(105):
  d0 = int(bytes0[-index - 1], 16)
  a1 = bytes1[-index - 1]
  a2 = bytes2[-index - 1]

  offset = index * 4
  # mask = (1 << (offset + 4)) - 1

  # print(a1, a2)

  x0 += d0 << offset

  # print(a1, a2)

  match a1, a2:
    case '?', '?':
      d0n = int(bytes0[-index - 2], 16)
      a1n = bytes1[-index - 2]
      a2n = bytes2[-index - 2]

      print('>>>', a1n, a2n)

      a = 0

      for d1 in range(16):
        for d2 in range(16):
          # print('test')
        #   x1p = x1 + (d1 << offset)
        #   x2p = x2 + (d2 << offset)

        #   if ((x1p * x2p) >> offset) & 0xf == d0:
        #     # x1 = x1p
        #     # x2 = x2p
        #     # break
        # else:
        #   continue

        # break

      # x1 = x1p
      # x2 = x2p
      # print('done', a)

          x1p = x1 + (d1 << offset)
          x2p = x2 + (d2 << offset)

          match a1n, a2n:
            case '?', _:
              d2n = int(a2n, 16)

              for d1n in range(16):
                if ((
                  (x1p + (d1n << (offset + 4))) *
                  (x2p + (d2n << (offset + 4)))
                ) >> offset) & 0xff == ((d0n << 4) + d0):
                  a += 1

                  if a == 0xf:
                    x1 = x1p
                    x2 = x2p
                    print('found', hex(d1), hex(d2), hex(d1n), hex(d2n))
                    break
              else:
                continue

              break

            case _, _:
              d1n = int(a1n, 16)
              d2n = int(a2n, 16)

              if ((
                (x1p + (d1n << (offset + 4))) *
                (x2p + (d2n << (offset + 4)))
              ) >> offset) & 0xff == ((d0n << 4) + d0):
                x1 = x1p
                x2 = x2p
                break
        else:
          continue

        break

    case _, '?':
      d1 = int(a1, 16)
      x1 += d1 << offset

      for d2 in range(16):
        x2p = x2 + (d2 << offset)

        if ((x1 * x2p) >> offset) & 0xf == d0:
          x2 = x2p
          break

    case '?', _:
      d2 = int(a2, 16)
      x2 += d2 << offset

      for d1 in range(16):
        x1p = x1 + (d1 << offset)

        # print(hex(((x1 * x2p) >> offset) & 0xf), hex(d0))

        if ((x2 * x1p) >> offset) & 0xf == d0:
          x1 = x1p
          break

    case _, _:
      d1 = int(a1, 16)
      d2 = int(a2, 16)

      x1 += d1 << offset
      x2 += d2 << offset

      # print(hex(x1), hex(x2))
      # print(hex((x1 * x2) & ((1 << (m + 4)) - 1)), hex(x0), hex((1 << (m + 4)) - 1))

      # rem += (d1 * d2 - x) >> 4
      # print(hex(d1), hex(d2), hex(rem))

# print(hex(x1))
# print(hex(x2))
print(hex(x1 * x2))
# print(hex(x0))


n = 0xcf358787ddd403a32e0a6b528cb5af567f26e5193f241355f63f21d1280e26bd110dcaed89cf23871ca4fe038f039c727956ca63055e068cb22fb73db06a699f352215a2fbf192f76ca598da3dc9f0d3bdc11ce2059d1e98939eb7538ace5288622635ca3806aabf159f5da8910e6116b78ace8bbd371bf02e5aa4345bcc9849f8343545660ab04e538c139f8419f9f51c11ec7616be689b196d8fbf16227b3ad7a42817936894149a1a1ad84e9d6ee6cefbb29fbc18fd8de93c4d9b18abebd5da4ee9f6484d0dae68ad2773c62553440790607d7c2c1c904a89a329d8da09708bf517436ffb36ae3f1e1931a78b0440a969726c3a88e9dc2b9c23987c4548b1
pq = x1 * x2

print(bin(n - pq))

print(f'p={x1:0x}')
print(f'q={x2:0x}')

# print(q)

  # print(hex(x), a1, a2)

  # match (a1, b1), (a2, b2):
  #   case ('?', _), (_, '?'):
  #     b1d = int(b1, 16)
  #     a2d = int(a2, 16)

  #     for a1d in range(16):
  #       for b2d in range(16):
  #         d1 = (a1d << 4) + b1d
  #         d2 = (a2d << 4) + b2d

  #         res = d1 * d2 + rem
  #         # print('>', hex(res & 0xff), hex(x))

  #         if (res & 0xff) == x:
  #           print('>', hex(d1), hex(d2), hex(rem))
  #           print(hex(x))

  #         # print(hex(d1), hex(d2))

  #   case (_, _), (_, _):
  #     d1 = int(c1, 16)
  #     d2 = int(c2, 16)

  #     rem += (d1 * d2 - x) >> 8
  #     # print(hex(rem))

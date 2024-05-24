import math
import random

import sympy
from sympy.ntheory import residue_ntheory


# g^q = 1 + kp
# p prime

#  p = kq + 1


a = 0x42fc44c0739a30729f87ce5d0f8c5e36f3fab7e1e9127fed8076bfbdd09bd4868f521d98f5431ceea93e0305d2bb49ffd73515b5551336ce3966f5c6b10e1230fc08fbcec82495988880bbd482581e1bb99040c17105357afe5de0c6e14ea8e1c5b73da462f5a0a6dc42ffe39366a48f4972e694bac646be99f73a3d984f9a6736bc7b7317cd03d303492b914f1a6b2de40fabc397b37fd6441533623a06c6b439f7de76d2281ee35ca6de63d6b219aad85923c77954e256eed0b0a10ebe55debebb73707291570dcc66e77fa05edc1a393fff98e35a9a5fd7eb1c2549f00a37fdc43e2e69284fa908cf9708c6ef5cae0e7c5ce8562d1d82c9c72e229b6efae2
b = a + 2 ** 1950
q = 0xef5a939beeedecb3081ea986a281f8dc7a51c968031d9cfb46b93f14d3a294e1


# p = sympy.randprime(a, b)
# p = 31879848417039132711033674580028535185714722059393957522842430359463716383051622802679816430567087329032806583925638317395717514134454532002619185978419725281701282953184763099360723043649877820287187815555509199237419961540878102702880135297212502483288889487372654363383849033533200355667400268860731712561378065633816159711539876581955466398802271494622514198400769400450267195677290559455763206313526205238992916326589222404751208606161324269991665375271782635964348722935092228670704751819999252004538485272151624693235719850914655577985949976675403055121233817625433787967695928158870973045519991437724712401549

# print(p)

# for g in range(2, 10000):
#   if pow(g, q, p) == 1:
#     print(g)
#     break
# else:
#   print('Failed')


# Ok
# s = a // q

# for k in range(s, s + 1000000):
#   p = k * q + 1

#   if sympy.isprime(p):
#     print(k - s)
#     break
# else:
#   print('fail')

# print(f'{k=}')
# print(f'{p=}')
# print(a <= p < b)


# k = 296917471318410758342387426763791182638962994532069527461574231072300074677756869786918491680598237863742139710940277456792151014473773790800510577463134280015273014788670543398208522464346577990884285233020755918046177975656919594966078683457679406297338027406171698703893910980568456052494017682116234433096251925862275797285659565498107932382553354434634716016487251613046568355558490368889413023043531489887322935030266505068751442937190598471514354283868367507453561744638275423445435053376516961040679976523202784746426474388383700832
# p = 31879848417039132711033674579979969033088945679717211949986499795443611115638490123995612400676076914873635053046558449879498907767882704373868624901780402026240114253768409585989183728015493641519308751007485475560523647446571353814075540577197767025734597944536475986946117175664093628106467932409991981250216097900851918555816347946203997999762461342012860298384808863021662599760370569856820297223525979928973205859266539525718442849736877428662381620797658538825521112132784669161475052222887547982748274705674278533171240295136237939748405147034222988229346075386923383967207949908118969415385873115352893163233

# print(sympy.isprime(k))

# for i in range(10):
#   print(pow(23453456506534568036450389475039403495790345, i, p))

  # print(residue_ntheory.is_primitive_root(2, p))

# for h in range(2, 10000):
#   # print(residue_ntheory.n_order(5, 301057))
#   # print(residue_ntheory.n_order(i, p))
#   # print(residue_ntheory.primitive_root(p))

#   if pow(h, k * q, p) == 1:
#     break

# print(2 ** k)
# g = h ** k



# # for i in range(0, 10000, 2):
# #   if is_prime(a + i):
# #     print(a + i)

# p = a + 1284
# # g = 345073945

# # assert pow(g, q, p) == 1

# # print(is_prime(p))
# print(p)
# print()
# print(sympy.totient(p))

# # for i in range(2, 10000):
# # for i in range(10000, 10000000):
# #   if i % 1000 == 0:
# #     print(i)

# #   if pow(i, q, p) == 1:
# #     print(i)
# #     break


# p = sympy.randprime(a, b)
# print(p)
# # g = primitive_root(p)

# print(residue_ntheory.n_order(10, p))



def isqrt(x: int):
    """Return the integer part of the square root of x, even for very
    large values."""
    if x < 0:
        raise ValueError('square root not defined for negative numbers')
    n = int(x)
    if n == 0:
        return 0
    a, b = divmod(n.bit_length(), 2)
    x = (1 << (a+b)) - 1
    while True:
        y = (x + n//x) // 2
        if y >= x:
            return x
        x = y

def find_invpow(x,n):
    """Finds the integer component of the n'th root of x,
    an integer such that y ** n <= x < (y + 1) ** n.
    """
    high = 1
    while high ** n <= x:
        high *= 2
    low = high//2
    while low < high:
        mid = (low + high) // 2
        if low < mid and mid**n < x:
            low = mid
        elif high > mid and mid**n > x:
            high = mid
        else:
            return mid
    return mid + 1


# https://math.stackexchange.com/questions/2951636/find-element-of-multiplicative-group-such-that-its-order-is-n
#
# Not necessarily efficient, but here is a suggestion:

# q must be a divisor of p - 1

# - find a multiple of q, kq such that kq + 1 is prime,
# - find a primitive element h of the cyclic group (𝐙/𝑝𝐙)× and
# - take g = h^k

# q = 761
# a = 100000
# b = 300000


# while True:
#   k = sympy.randprime(isqrt(a // q // 2), isqrt(b // q // 2))
#   p = k ** 2 * q * 2 + 1

#   if sympy.isprime(p):
#     assert a <= p < b
#     print(f'{k=}')
#     break

# k = 287681838580426927283216823921510424598031568852493557319788440395575247829003839504777544450607061286609499153920694667864164665316190518100573953633546736452493543997597444641403784270214573206263813329346190092342247130695937115812676813212455534672463865617057876647
k = 197620091398096086065199528233873212564881218769356721966766325035966285939073843971286827307438367760395819632548471839581925424887928882470032744092869264600713510375962048283067186036966824544392687920172335973858650445138950952696799531611707264073264216273029282387
l = k ** 2 * 2
pm1 = k ** 2 * q * 2
p = pm1 + 1

factors = [k, 2, q]

for a in range(2, 100):
  for factor in factors:
    if pow(a, pm1 // factor, p) == 1:
       break
  else:
    print(a)
    break

g = pow(2, l, p)

print(a <= p < b)
print(f'{p=}')
print(f'{g=}')

print(pow(g, q, p))

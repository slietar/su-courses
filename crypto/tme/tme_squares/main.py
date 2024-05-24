# https://math.stackexchange.com/questions/5877/efficiently-finding-two-squares-which-sum-to-a-prime/5883#5883

def mods(a, n):
    if n <= 0:
        return "negative modulus"
    a = a % n
    if (2 * a > n):
        a -= n
    return a

def powmods(a, r, n):
    out = 1
    while r > 0:
        if (r % 2) == 1:
            r -= 1
            out = mods(out * a, n)
        r //= 2
        a = mods(a * a, n)
    return out

def quos(a, n):
    if n <= 0:
        return "negative modulus"
    return (a - mods(a, n))//n

def grem(w, z):
    # remainder in Gaussian integers when dividing w by z
    (w0, w1) = w
    (z0, z1) = z
    n = z0 * z0 + z1 * z1
    if n == 0:
        return "division by zero"
    u0 = quos(w0 * z0 + w1 * z1, n)
    u1 = quos(w1 * z0 - w0 * z1, n)
    return(w0 - z0 * u0 + z1 * u1,
           w1 - z0 * u1 - z1 * u0)

def ggcd(w, z):
    while z != (0,0):
        w, z = z, grem(w, z)
    return w

def root4(p):
    # 4th root of 1 modulo p
    if p <= 1:
        return "too small"
    if (p % 4) != 1:
        return "not congruent to 1"
    k = p//4
    j = 2
    while True:
        a = powmods(j, k, p)
        b = mods(a * a, p)
        if b == -1:
            return a
        if b != 1:
            return "not prime"
        j += 1

def sq2(p):
    a = root4(p)
    return ggcd((p,0),(a,1))

p = 0xf0d724a016975b540cd486238e777b09c559d6e59bd933659834fbdfdea356392164a6b994e22382e4ba61c1baae26aa8bee5a1752857899a7a6c774fa34c4ede6bcc6aa0a923d36ceb597312512e879d26cd69a50b9b8de7cdc24c1749e8061bbfa6021437051b80d844930b5f96bb5ab9ef1e697ec1aa96a224c18b660f0c9
a, b = sq2(p)

a = abs(a)
b = abs(b)

assert a ** 2 + b ** 2 == p
print(f'{a=}')
print(f'{b=}')

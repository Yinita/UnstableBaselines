import torch, math, torch.nn.functional as F
torch.manual_seed(0)

d, V = 4, 5                 # hidden dim & vocab size
E  = torch.randn(V, d, requires_grad=True)
Wq = torch.randn(d, d, requires_grad=True)
Wk = torch.randn(d, d, requires_grad=True)   # <-- prefix-only param
Wv = torch.randn(d, d, requires_grad=True)
Wo = torch.randn(d, d, requires_grad=True)
params = [E, Wq, Wk, Wv, Wo]

ids = torch.tensor([1, 2])  # two-token sequence
target = torch.tensor([3])  # predict next token for position 1

def loss_full():
    h = E[ids]                        # (2,d)
    K, Vv        = h @ Wk, h @ Wv
    q2           = (h[1] @ Wq).unsqueeze(0)     # (1,d)
    score        = (q2 @ K[:1].T) / math.sqrt(d)
    w            = F.softmax(score, dim=-1)      # attend only token 0
    ctx          = w @ Vv[:1]                   # (1,d)
    logits       = ctx @ Wo @ E.T               # (1,V)
    return F.cross_entropy(logits, target)

def loss_chunk():
    with torch.no_grad():              # prefix in no-grad
        k0 = (E[ids[0]] @ Wk).detach()
        v0 = (E[ids[0]] @ Wv).detach()
    h1  = E[ids[1]]                    # requires_grad
    q2  = h1 @ Wq
    score = (q2 @ k0) / math.sqrt(d)   # uses detached k0
    w     = torch.softmax(score.unsqueeze(0), dim=-1)
    ctx   = w @ v0.unsqueeze(0)        # detached v0
    logits= (ctx @ Wo) @ E.T
    return F.cross_entropy(logits, target)

# ---------- run full pass ----------
for p in params: p.grad = None
loss_full().backward()
grad_full = Wk.grad.clone()

# ---------- run chunked pass ----------
for p in params: p.grad = None
loss_chunk().backward()
grad_chunk = Wk.grad.clone()

print("‖grad_full‖  =", grad_full.norm().item())
print("‖grad_chunk‖ =", grad_chunk.norm().item())

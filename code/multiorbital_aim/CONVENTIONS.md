# 多轨道 AIM 的 MPS-DQME 冻结约定

## 1. DQME

\[
\dot{\boldsymbol\rho}=-i[H_{\rm sys},\boldsymbol\rho]
-\sum_{\alpha\nu sk}\left(\gamma^-_{\alpha\nu sk}\hat N_{\alpha\nu sk}\boldsymbol\rho
+\gamma^+_{\alpha\nu sk}\boldsymbol\rho\hat N_{\alpha\nu sk}\right)+\text{fermionic ladder couplings}.
\]

环境指标为 lead `alpha`、orbital `nu`、spin `s`、pole `k` 和 sign `+/-`。每个
`(alpha,nu,s)` 通道拥有一组费米 dissipatons；当前不含跨轨道浴相关。保留
`Absolute_full_DQME` 已验证的 `zeta/ksi` 系数、费米 ladder 项和电流公式。

## 2. 体系 Hamiltonian

\[
\begin{aligned}
H_{\rm sys}={}&\sum_{mm'\sigma}\epsilon_{mm'}d^\dagger_{m\sigma}d_{m'\sigma}
+U\sum_m n_{m\uparrow}n_{m\downarrow}\\
&+U'\sum_{m<m',\sigma}n_{m\sigma}n_{m'\bar\sigma}
+(U'-J_1)\sum_{m<m',\sigma}n_{m\sigma}n_{m'\sigma}\\
&-J_2\sum_{m\ne m'}d^\dagger_{m\uparrow}d_{m\downarrow}
d^\dagger_{m'\downarrow}d_{m'\uparrow}\\
&+J_3\sum_{m\ne m'}d^\dagger_{m\uparrow}d^\dagger_{m\downarrow}
d_{m'\downarrow}d_{m'\uparrow}.
\end{aligned}
\]

`m != m'` 是有序轨道对求和，不能擅自替换成 `m < m'`。`epsilon` 必须为
Hermitian。当前默认 `Uprime=J1=J2=J3=0`，但代码完整保留这些项。

## 3. 基组与 MPS 链

体系 spin-orbital 的逻辑顺序为
`(0,up),(0,down),(1,up),(1,down),...`，局域基为 `|0>,|1>`。

链顺序为：

`system-ket -> dissipaton-ket(m) -> dissipaton-bra(n, reversed) -> system-bra(reversed)`。

ket 的第一个 spin-orbital 是左端位，其镜像 bra site 是右端位；其他 sites
均属于 `vmid`。因此 TDVP 的两端加中间链接口不变。

## 4. Jordan-Wigner 与 bra 镜像

- ket system 和 ket dissipatons 的 J-W string 锚定左边界；
- bra dissipatons 和 bra system 的 J-W string 锚定右边界；
- bra 物理标签和 site 顺序严格镜像 ket；
- 右作用使用转置，乘积转置时反转算符因子次序；
- 所有体系与环境费米算符属于同一反对易代数。

电流读取保留
`rho_m_phys = P @ rho_m_raw`、`rho_n_phys = rho_n_raw @ P`，其中
\(P=(-1)^{\sum_{m,s}n_{ms}}\)。DQME 与既有 HEOM 比较采用既定的反向电流符号。

## 5. 验证顺序

1. `nvarf=1` 且所有轨道间参数为零，回归普通 SIAM。
2. product-MPO Hamiltonian 与同基组 dense Hamiltonian 逐项比较。
3. 检查 ket/bra 镜像、Hermiticity、真空 RDO 和零初始电流。
4. 短时检查 trace、真正中心差分连续性和四路电流。
5. `nvarf=2` 后逐项开启 hopping、`Uprime/J1/J2/J3`，与整体体系 MPS-HEOM benchmark 对照。

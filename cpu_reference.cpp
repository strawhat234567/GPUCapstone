#include "attention.cu"
#include <cmath>
#include <algorithm>

void cpu_attention(
    const std::vector<float> &q,
    const std::vector<float> &k,
    const std::vector<float> &v,
    std::vector<float> &o,
    int n,
    int d)
{
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> p(n);

    for (int i = 0; i < n; ++i)
    {
        float m = -INFINITY;

        for (int j = 0; j <= i; ++j)
        {
            float a = 0.0f;
            for (int x = 0; x < d; ++x)
            {
                a += q[i * d + x] * k[j * d + x];
            }
            p[j] = a * scale;
            m = std::max(m, p[j]);
        }

        float z = 0.0f;
        for (int j = 0; j <= i; ++j)
        {
            p[j] = std::exp(p[j] - m);
            z += p[j];
        }

        for (int x = 0; x < d; ++x)
        {
            float acc = 0.0f;
            for (int j = 0; j <= i; ++j)
            {
                acc += (p[j] / z) * v[j * d + x];
            }
            o[i * d + x] = acc;
        }
    }
}
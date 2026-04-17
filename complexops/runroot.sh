./math/sqrtrsq_test --dump-inputs ./roottest.in
../bin/torchunary.py --op sqrt --file ./roottest.in
../bin/torchunary.py --op rsqrt --file ./roottest.in
./math/sqrtrsq_test --op sqrt --torchinductor torchinductorsqrt.bin --torcheager torcheagersqrt.bin --verbose --quiet --color | less -R
./math/sqrtrsq_test --op rsqrt --torchinductor torchinductorrsqrt.bin --torcheager torcheagerrsqrt.bin --verbose --quiet --color | less -R

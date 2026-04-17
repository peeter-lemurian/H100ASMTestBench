./math/cossin_test --dump-inputs ./trigunarytest.in
../bin/torchunary.py --op sin --file ./trigunarytest.in
../bin/torchunary.py --op cos --file ./trigunarytest.in
./math/cossin_test --op sin --torchinductor torchinductorsin.bin --torcheager torcheagersin.bin --verbose --quiet --color | less -R
./math/cossin_test --op cos --torchinductor torchinductorcos.bin --torcheager torcheagercos.bin --verbose --quiet --color | less -R

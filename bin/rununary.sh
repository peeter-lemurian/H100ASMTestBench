complexops/unary_test --dump-inputs ./unarytest.in
for i in absf acosf acoshf asinf asinhf atanf atanhf cbrtf ceilf cosf coshf cospif erfcf erff exp10f exp2f expf expm1f floorf lgammaf log10f log1pf log2f logbf logf modff-frac modff-int nearbyintf reciprocal roundf rsqrtf sinf sinhf sinpif sqrtf tanf tanhf tanpif tgammaf truncf ; do
#for i in exp10f ; do
../bin/torchunary.py --op $i --file ./unarytest.in
done
#exit
for i in absf acosf acoshf asinf asinhf atanf atanhf cbrtf ceilf cosf coshf cospif erfcf erff exp10f exp2f expf expm1f floorf lgammaf log10f log1pf log2f logbf logf modff-frac modff-int nearbyintf reciprocal roundf rsqrtf sinf sinhf sinpif sqrtf tanf tanhf tanpif tgammaf truncf ; do
complexops/unary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --csv > $i.csv
complexops/unary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color | tee $i.out
#complexops/unary_test --$i --torchinductor torchinductor$i.bin --torcheager torcheager$i.bin --quiet --color | less -R
done

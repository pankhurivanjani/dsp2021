%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Tutorial zur Vorlesung "Digitale Signalverarbeitung 2010"
%%%%  Autor: Munir Georges
%%%%  Datum: 03.2010
%%%%  EMail: Munir.Georges@lsv.uni-saarland.de
%%%%  EMail: Georges@globalinventor.eu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = blockproc(img, b, fun)

[y,x] = size(img);
by = b(1);
bx = b(2);

out = zeros(y,x);
for i=1:bx:x
    for j=1:by:y
        %size(img(j:j+by-1,i:i+bx-1))
        out(j:j+by-1,i:i+bx-1) = fun(img(j:j+by-1,i:i+bx-1));
    end
end

%Example:
%DCT
%input img
%qua = quantisierung
%res = DCT of size(bx,by) of img
%output res with img = res

%JPEG
%out = blockproc(img,[10 10], @dct2);
%a = quantisierungsmatrix
%fun = @(b) b.*a
%qua = blockproc(out,[10 10], fun);
%res = blockproc(qua,[10 10], @idct2);
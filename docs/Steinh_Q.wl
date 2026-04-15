
(* !/usr/bin/env wolframscript *)


# 以下是dx,dy,dz生成

(* 
(* 初始化环境 *)
ClearAll["Global`*"];

$Assumptions = {dx \[Element] Reals, dy \[Element] Reals, dz \[Element] Reals, r > 0};

(* 辅助函数：将表达式转换为 C++ 宏风格字符串 *)
ToCodeString[expr_] := Module[{s},
    s = expr;
    (* 变量替换 *)
    s = s /. {x -> dx, y -> dy, z -> dz};
    (* 幂次项模式匹配 *)
    s = s /. {
        Power[base_, 8] :> "POW8(" <> ToString[base, InputForm] <> ")",
        Power[base_, 7] :> "POW7(" <> ToString[base, InputForm] <> ")",
        Power[base_, 6] :> "POW6(" <> ToString[base, InputForm] <> ")",
        Power[base_, 5] :> "POW5(" <> ToString[base, InputForm] <> ")",
        Power[base_, 4] :> "POW4(" <> ToString[base, InputForm] <> ")",
        Power[base_, 3] :> "POW3(" <> ToString[base, InputForm] <> ")",
        Power[base_, 2] :> "POW2(" <> ToString[base, InputForm] <> ")"
    };
    s = ToString[s, InputForm];
    s = StringReplace[s, {"\"" -> "", " " -> ""}];
    s
];

(* 辅助函数：将公因式表达式转换为 C++ 宏 *)
GenerateGlobalCCode[globalCommonC_, label_] := Module[{cPart, vPart, num, den, numericStr, numStr, denStr, replaceRules, finalCode},
    cPart = Select[Factor[globalCommonC], FreeQ[#, x | y | z] &];
    vPart = globalCommonC/cPart;
    If[NumberQ[N[cPart]] && N[cPart] < 0, cPart = -cPart; vPart = -vPart;];
    numericStr = ToString[CForm[N[cPart, 32]]];
    
    replaceRules = {
        (x^2 + y^2 + z^2)^8 -> "POW8(r2)", (x^2 + y^2 + z^2)^7 -> "POW7(r2)",
        (x^2 + y^2 + z^2)^6 -> "POW6(r2)", (x^2 + y^2 + z^2)^5 -> "POW5(r2)",
        (x^2 + y^2 + z^2)^4 -> "POW4(r2)", (x^2 + y^2 + z^2)^3 -> "POW3(r2)",
        (x^2 + y^2 + z^2)^2 -> "POW2(r2)", (x^2 + y^2 + z^2) -> "r2",
        x^7 -> "POW7(dx)", y^7 -> "POW7(dy)", z^7 -> "POW7(dz)",
        x^6 -> "POW6(dx)", y^6 -> "POW6(dy)", z^6 -> "POW6(dz)",
        x^5 -> "POW5(dx)", y^5 -> "POW5(dy)", z^5 -> "POW5(dz)",
        x^4 -> "POW4(dx)", y^4 -> "POW4(dy)", z^4 -> "POW4(dz)",
        x^3 -> "POW3(dx)", y^3 -> "POW3(dy)", z^3 -> "POW3(dz)",
        x^2 -> "POW2(dx)", y^2 -> "POW2(dy)", z^2 -> "POW2(dz)",
        x -> dx, y -> dy, z -> dz
    };

    num = Numerator[vPart]; den = Denominator[vPart];
    numStr = StringReplace[ToString[num /. replaceRules, InputForm], {"\"" -> "", " " -> ""}];
    denStr = StringReplace[ToString[den /. replaceRules, InputForm], {"\"" -> "", " " -> ""}];
    
    finalCode = "temp = " <> numericStr <> "*(" <> numStr <> ")/(" <> denStr <> ");";
    Print["/* Final C++ Code for " <> label <> " */"];
    Print[finalCode];
    finalCode
];

(* 内部求导函数 *)
InternalGetRawDerivative[targetFunc_, var_] := Module[{A, dRe, dIm, togRe, togIm, finalAlgebraicRule},
    A = FullSimplify[D[targetFunc, var]];
    dRe = ComplexExpand[Re[A], {x, y, z} \[Element] Reals && r > 0];
    dIm = ComplexExpand[Im[A], {x, y, z} \[Element] Reals && r > 0];
    
    finalAlgebraicRule = {
        Cos[m_. p_] /; (!FreeQ[p, ArcTan] || !FreeQ[p, phiA]) :> Re[Expand[(x + I y)^m]]/(x^2 + y^2)^(m/2),
        Sin[m_. p_] /; (!FreeQ[p, ArcTan] || !FreeQ[p, phiA]) :> Im[Expand[(x + I y)^m]]/(x^2 + y^2)^(m/2),
        Re[expr_] :> expr, Im[expr_] :> 0
    };
    
    togRe = Together[Simplify[dRe /. finalAlgebraicRule]];
    togIm = Together[Simplify[dIm /. finalAlgebraicRule]];
    {Numerator[togRe], Numerator[togIm], Denominator[togRe]}
];

(* 核心处理函数 *)
ProcessSphericalDerivativePartialAll[targetFunc_, label_] := Module[{resX, resY, resZ, allNumerators, globalGCD, globalDen, globalCommonC, directions},
    Print["\n" <> StringRepeat["=", 20] <> " " <> label <> " " <> StringRepeat["=", 20]];
    
    resX = InternalGetRawDerivative[targetFunc, x];
    resY = InternalGetRawDerivative[targetFunc, y];
    resZ = InternalGetRawDerivative[targetFunc, z];
    
    allNumerators = {resX[[1]], resX[[2]], resY[[1]], resY[[2]], resZ[[1]], resZ[[2]]};
    globalGCD = PolynomialGCD @@ allNumerators;
    globalDen = resX[[3]];
    globalCommonC = globalGCD/globalDen;

    GenerateGlobalCCode[globalCommonC, label];
    
    directions = {{"Dx", resX}, {"Dy", resY}, {"Dz", resZ}};
    Do[
        dirName = d[[1]];
        Module[{bReal, bImag},
            bReal = FullSimplify[d[[2, 1]]/globalGCD];
            bImag = FullSimplify[d[[2, 2]]/globalGCD];
            Print["//--- Direction: ", d[[1]], " ---"];
            If[bReal =!= 0, Print["t", dirName, "_r = temp*(", ToCodeString[bReal], ");"]];
            If[bImag =!= 0, Print["t", dirName, "_i = temp*(", ToCodeString[bImag], ");"]];
        ], {d, directions}
    ];
];

(* 定义物理坐标 *)
r = Sqrt[x^2 + y^2 + z^2];
thetaA = ArcCos[z/r];
phiA = ArcTan[x, y];

(* 主循环：执行 l=6 系列 *)
mList = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6};
Do[
    ProcessSphericalDerivativePartialAll[
        SphericalHarmonicY[6, m, thetaA, phiA], 
        "Y6" <> If[m < 0, "n" <> ToString[Abs[m]], ToString[m]]
    ], 
    {m, mList}
]; *)
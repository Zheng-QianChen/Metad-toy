
(* !/usr/bin/env wolframscript *)

(* 初始化环境 *)
ClearAll["Global`*"];

$Assumptions = {dx \[Element] Reals, dy \[Element] Reals, dz \[Element] Reals, r > 0};ClearAll[ProcessSphericalDerivativePartialAll];

$Assumptions = {x \[Element] Reals, y \[Element] Reals, 
   z \[Element] Reals, r > 0};





(*辅助函数：将表达式转换为 C++ 宏风格字符串*)

ToCodeString[expr_] := Module[{s}, s = expr;
  (*1. 先把基础变量替换掉，防止 ToString 乱码*)s = s /. {x -> dx, y -> dy, z -> dz};
  (*2. 使用专门的模式匹配处理所有幂次项，无论底数是什么*)(*这样 (dx^2+dy^2)^2 会被正确识别*)s = s /. {
     Power[base_, 7] :> "POW5(" <> ToString[base, InputForm] <> ")",
     Power[base_, 6] :> "POW5(" <> ToString[base, InputForm] <> ")",
     Power[base_, 5] :> "POW5(" <> ToString[base, InputForm] <> ")", 
     Power[base_, 4] :> "POW4(" <> ToString[base, InputForm] <> ")", 
     Power[base_, 3] :> "POW3(" <> ToString[base, InputForm] <> ")", 
     Power[base_, 2] :> "POW2( " <> ToString[base, InputForm] <> ")"};
  (*3. 最终转为字符串并清理*)s = ToString[s, InputForm];
  s = StringReplace[
    s, {"\"" -> "", " " -> "", "dx" -> "dx", "dy" -> "dy", 
     "dz" -> "dz"}];
  (*4. 这里的二次清理是为了防止 POW 内部还有未转换的^符号*)s = StringReplace[s,
    {"dx^2" -> "POW2(dx)", "dy^2" -> "POW2(dy)", "dz^2" -> "POW2(dz)",
      "dx^3" -> "POW3(dx)", "dy^3" -> "POW3(dy)", "dz^3" -> "POW3(dz)",
      "dx^4" -> "POW4(dx)", "dy^4" -> "POW4(dy)", "dz^4" -> "POW4(dz)",
      "dx^5" -> "POW5(dx)", "dy^5" -> "POW5(dy)", "dz^5" -> "POW5(dz)",
      "dx^6" -> "POW6(dx)", "dy^6" -> "POW6(dy)", "dz^6" -> "POW6(dz)",
      "dx^7" -> "POW7(dx)", "dy^7" -> "POW7(dy)", "dz^7" -> "POW7(dz)"}];
  s]



(*辅助函数：将公因式表达式转换为 C++ 宏风格字符串*)

GenerateGlobalCCode[globalCommonC_, label_] := 
 Module[{cPart, vPart, num, den, numericStr, numStr, denStr, 
   finalCode, replaceRules},(*1. 分离常数项*)
  cPart = Select[Factor[globalCommonC], FreeQ[#, x | y | z] &];
  vPart = globalCommonC/cPart;
  (*2. 强制常数项为正，负号留给坐标部分*)
  If[NumberQ[N[cPart]] && N[cPart] < 0, cPart = -cPart;
   vPart = -vPart;];
  numericStr = ToString[CForm[N[cPart, 32]]];
  (*3. 定义精准替换规则库 (包含 3 次方和 4 次方)*)(*注意：规则顺序很重要，先匹配整体分母，再匹配单个分量*)
  replaceRules = {
    (x^2 + y^2 + z^2)^8 -> "POW8(r2)",
    (x^2 + y^2 + z^2)^7 -> "POW7(r2)",
    (x^2 + y^2 + z^2)^6 -> "POW6(r2)",
    (x^2 + y^2 + z^2)^5 -> "POW5(r2)",
    (x^2 + y^2 + z^2)^4 -> "POW4(r2)",
    (x^2 + y^2 + z^2)^3 -> "POW3(r2)",
    (x^2 + y^2 + z^2)^2 -> "POW2(r2)",
    (x^2 + y^2 + z^2) -> "r2",
    x^7 -> "POW7(dx)", y^7 -> "POW7(dy)", z^7 -> "POW7(dz)",
    x^6 -> "POW6(dx)", y^6 -> "POW6(dy)", z^6 -> "POW6(dz)",
    x^5 -> "POW5(dx)", y^5 -> "POW5(dy)", z^5 -> "POW5(dz)", 
    x^4 -> "POW4(dx)", y^4 -> "POW4(dy)", z^4 -> "POW4(dz)",
    x^3 -> "POW3(dx)", y^3 -> "POW3(dy)", z^3 -> "POW3(dz)",
    x^2 -> "POW2(dx)", y^2 -> "POW2(dy)", z^2 -> "POW2(dz)",
    x -> dx, y -> dy, z -> dz};
  (*4. 拆分分子分母分别替换，防止 Mathematica 自动约分破坏结构*)num = Numerator[vPart];
  den = Denominator[vPart];
  numStr = 
   StringReplace[
    ToString[num /. replaceRules, InputForm], {"\"" -> "", " " -> ""}];
  denStr = 
   StringReplace[
    ToString[den /. replaceRules, InputForm], {"\"" -> "", " " -> ""}];
  (*5. 拼接最终 C++ 代码*)
  finalCode = 
   "temp = " <> numericStr <> "*(" <> numStr <> ")/(" <> denStr <> ");";
  Print[Style["/* Final C++ Code for " <> label <> " */", DarkGray, 
    Italic]];
  Print[finalCode];
  finalCode]



(*核心计算函数*)



ProcessSphericalDerivativePartialAll[targetFunc_, label_] := 
  Module[{resX, resY, resZ, allNumerators, globalGCD, globalDen, 
    globalCommonC, cleanResX, cleanResY, cleanResZ, directions}, 
   Print[Style[
     Row[{"\n====================== Analyzing ALL partials for ", label,
        " ========================="}], Green, Bold]];
   (*1. 获取三个方向的原始分子和分母*)
   resX = InternalGetRawDerivative[targetFunc, x];
   resY = InternalGetRawDerivative[targetFunc, y];
   resZ = InternalGetRawDerivative[targetFunc, z];
   (*2. 提取全局公因子*)(*收集所有 6 个分子:{numReX,numImX,numReY,numImY,numReZ,
   numImZ}*)allNumerators = {resX[[1]], resX[[2]], resY[[1]], resY[[2]],
      resZ[[1]], resZ[[2]]};
   globalGCD = PolynomialGCD @@ allNumerators;
   (*假设分母在 x,y,z 方向通常是相同的 (r的幂次)，取其中一个即可*)globalDen = resX[[3]];
   globalCommonC = globalGCD/globalDen;
   Print[
    Style[">>> Global Common Factor (x,y,z mixed) <<<", Bold, Red]];
   Print["Global C = ", globalCommonC];
   Print["Numeric (32-bit): ", 
    Abs[N[Select[Factor[globalCommonC], FreeQ[#, x | y | z] &], 32]]];
   (*调用更新后的生成函数*)GenerateGlobalCCode[globalCommonC, label];
   (*3. 打印结果并输出代码格式*)
   directions = {{"Dx", resX}, {"Dy", resY}, {"Dz", resZ}};
   Do[Module[{bReal, bImag, codeReal}, 
     bReal = FullSimplify[d[[2, 1]]/globalGCD];
     bImag = FullSimplify[d[[2, 2]]/globalGCD];
     Print[
      Style[Row[{"--- Result for ", label, "_", d[[1]], " ---"}], 
       Blue]];
     Print["Bracket Real = ", bReal];
     (*如果实部不为 0，生成代码行*)If[bReal =!= 0, codeReal = ToCodeString[bReal];
      Print[Style["C++ Code: ", Gray], "temp*(", codeReal, ");"];];
     If[bImag =!= 0, Print["Bracket Imag = ", bImag];
      Print[Style["C++ Code Imag: ", Gray], "temp*(", 
       ToCodeString[bImag], ");"];];], {d, directions}];
   {globalCommonC, label}];



(*辅助函数：只负责计算和初级通分，不进行最终化简*)
InternalGetRawDerivative[targetFunc_, var_] := 
  Module[{A, dRe, dIm, togRe, togIm}, 
   A = FullSimplify[D[targetFunc, var], 
     Assumptions -> {x, y, z} \[Element] Reals];
   dRe = ComplexExpand[Re[A], {x, y, z} \[Element] Reals && r > 0];
   dIm = ComplexExpand[Im[A], {x, y, z} \[Element] Reals && r > 0];
   togRe = Together[dRe];
   togIm = Together[dIm];
   finalAlgebraicRule = {Cos[
        m_. p_] /; (! FreeQ[p, ArcTan] || ! FreeQ[p, Arg] || ! 
          FreeQ[p, phiA]) :> Re[Expand[(x + I y)^m]]/(x^2 + y^2)^(m/2),
      Sin[m_. p_] /; (! FreeQ[p, ArcTan] || ! FreeQ[p, Arg] || ! 
          FreeQ[p, phiA]) :> 
      Im[Expand[(x + I y)^m]]/(x^2 + y^2)^(m/2),(*彻底移除任何残留的 Re/Im 包装*)
     Re[expr_] :> expr, Im[expr_] :> 0};
   togRe = Simplify[togRe /. finalAlgebraicRule];
   togIm = Simplify[togIm /. finalAlgebraicRule];
   {Numerator[togRe], Numerator[togIm], Denominator[togRe]}];
(*InternalGetRawDerivative[targetFunc_,var_]:=Module[{A,dRe,dIm,togRe,\
togIm,algebraRule,cleanTarget},(*在求导前强制实数化目标函数，消除潜在的 Im[x]*)\
cleanTarget=ComplexExpand[targetFunc,{x,y,z},TargetFunctions->{Re,Im}];
A=D[cleanTarget,var];
(*代数规则：Cos[m phi]->代数式*)algebraRule={Cos[m_. p_]/;(!FreeQ[p,Arg]||\
!FreeQ[p,ArcTan]||!FreeQ[p,phiA]):>Re[Expand[(x+I y)^m]]/(x^2+y^2)^(m/\
2),Sin[m_. p_]/;(!FreeQ[p,Arg]||!FreeQ[p,ArcTan]||!FreeQ[p,phiA]):>Im[\
Expand[(x+I y)^m]]/(x^2+y^2)^(m/2)};
A=A/. algebraRule;
(*再次强制实数化提取*)dRe=ComplexExpand[Re[A],{x,y,z},TargetFunctions->{Re,Im}]//\
Expand;
dIm=ComplexExpand[Im[A],{x,y,z},TargetFunctions->{Re,Im}]//Expand;
togRe=Together[dRe];
togIm=Together[dIm];
{Numerator[togRe],Numerator[togIm],If[togRe===0,Denominator[togIm],\
Denominator[togRe]]}];*)

ConvertToAlgebraic[targetFunc_, label_] := Module[
    {resX, allNums, globalGCD, globalDen, globalCommonC, bReal, bImag},
    
    Print["\n" <> StringRepeat["=", 15] <> " Processing Ylm: " <> label <> " " <> StringRepeat["=", 15]];
    
    (* 1. 借用 InternalGetRawDerivative 获取代数化的分子和分母 *)
    (* 虽然名字叫 Derivative，但它内部的 finalAlgebraicRule 正好是我们需要的坐标转换 *)
    resX = InternalGetRawDerivative[targetFunc, x]; 
    (* 注意：如果 targetFunc 已经是代数式而非导数，resX 这里的 '求导' 动作可能多余 *)
    (* 更好的做法是直接对 targetFunc 应用转换规则，如下：*)
    
    {bRealRaw, bImagRaw, globalDen} = InternalGetRawDerivative[targetFunc, x];
    
    (* 2. 提取分子间的公因子 *)
    allNums = {bRealRaw, bImagRaw};
    globalGCD = PolynomialGCD @@ allNums;
    globalCommonC = globalGCD / globalDen;

    (* 3. 输出 temp 变量 (包含数值系数和 r 的幂次分母) *)
    GenerateGlobalCCode[globalCommonC, label];
    
    (* 4. 输出最终的实部和虚部赋值 *)
    bReal = FullSimplify[bRealRaw / globalGCD];
    bImag = FullSimplify[bImagRaw / globalGCD];
    
    Print["//--- Cartesian Ylm Result ---"];
    If[bReal =!= 0, 
        Print[label, "_r = temp * (", ToCodeString[bReal], ");"],
        Print[label, "_r = 0.0;"]
    ];
    If[bImag =!= 0, 
        Print[label, "_i = temp * (", ToCodeString[bImag], ");"],
        Print[label, "_i = 0.0;"]
    ];
];


(* 定义物理坐标 *)
r = Sqrt[x^2 + y^2 + z^2];
thetaA = ArcCos[z/r];
phiA = ArcTan[x, y];

(* 主循环：执行 l=6 系列 *)
mList = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6};
Do[
    ConvertToAlgebraic[
        SphericalHarmonicY[6, m, thetaA, phiA],
        "Y6" <> If[m < 0, "n" <> ToString[Abs[m]], ToString[m]]
    ], 
    {m, mList}
];
Do[
    ProcessSphericalDerivativePartialAll[
        SphericalHarmonicY[6, m, thetaA, phiA], 
        "Y6" <> If[m < 0, "n" <> ToString[Abs[m]], ToString[m]]
    ], 
    {m, mList}
];
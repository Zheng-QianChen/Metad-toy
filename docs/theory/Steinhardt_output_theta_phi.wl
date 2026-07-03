
(* !/usr/bin/env wolframscript *)

(* 1. 设置全局假设，确保 Re/Im 不会产生冗余保护 *)
$Assumptions = {thetaB \[Element] Reals, phiB \[Element] Reals, thetaB > 0, thetaB < Pi};

(* 2. 定义核心提取函数 *)
ExtractYlmParts[l_, m_] := Module[
  {ylm, ce, re, im, commonFactor, finalRe, finalIm},
  
  (* 获取标准球谐函数并展开 *)
  ylm = SphericalHarmonicY[l, m, thetaB, phiB];
  ce = ComplexExpand[ylm];
  
  (* 提取实部和虚部 *)
  re = FullSimplify[Re[ce]];
  im = FullSimplify[Im[ce]];
  
  (* 3. 提取公因式 (幅度部分) *)
  (* 逻辑：寻找实部中 Cos[m*phiB] 的系数，或虚部中 Sin[m*phiB] 的系数 *)
  If[m == 0,
    commonFactor = re;
    finalRe = 1;
    finalIm = 0;,
    (* 对于 m > 0, 提取共同的 theta 依赖项 *)
    commonFactor = FullSimplify[re / Cos[m*phiB]];
    finalRe = Cos[m*phiB];
    finalIm = Sin[m*phiB];
  ];
  
  (* 4. 返回结果列表 *)
  {commonFactor, finalRe, finalIm}
];

GenerateCppCode[l_, m_] := Module[
  {parts, common, rePart, imPart, label, commonStr, reStr, imStr, formatStr},
  
  (* 内部定义格式化函数 *)
  formatStr[expr_] := Module[{s, rules},
    s = ToString[CForm[N[expr, 16]]];
    rules = {
      "Power(" ~~ base : Shortest[__] ~~ "," ~~ exp : DigitCharacter .. ~~ ")" :> 
        "POW" <> exp <> "(" <> base <> ")",
      "Cos(thetaB)" | "Cos(1.*thetaB)" -> "cos_theta",
      "Sin(thetaB)" | "Sin(1.*thetaB)" -> "sin_theta",
      "Cos(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "thetaB)" :> "cos_" <> d <> "theta",
      "Sin(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "thetaB)" :> "sin_" <> d <> "theta",
      "Cos(phiB)" | "Cos(1.*phiB)" -> "cos_phi",
      "Sin(phiB)" | "Sin(1.*phiB)" -> "sin_phi",
      "Cos(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "phiB)" :> "cos_" <> d <> "phi",
      "Sin(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "phiB)" :> "sin_" <> d <> "phi",
      "Abs(" ~~ x__ ~~ ")" :> x, (* 清理可能的 Abs *)
      " " -> ""
    };
    FixedPoint[StringReplace[#, rules] &, s]
  ];

  parts = ExtractYlmParts[l, m];
  common = parts[[1]];
  rePart = parts[[2]];
  imPart = parts[[3]];
  label = ToString[l] <> "," <> ToString[m];

  commonStr = formatStr[common];
  reStr = formatStr[rePart];
  imStr = formatStr[imPart];

  Print["// Y,", label];
  If[m > 0, Print["// Y,", l, ",", -m, ", -Re+Im"]];
  Print["temp_value = ", commonStr, ";"];
  
  If[m == 0,
    Print["d_stein_Ylm[stein_Ylm_base_id + 0] = temp_value;"];
    Print["d_stein_Ylm[stein_Ylm_base_id + 1] = 0.0;"],
    Print["d_stein_Ylm[stein_Ylm_base_id + ", Abs[m]*2, "] = temp_value * ", reStr, ";"];
    Print["d_stein_Ylm[stein_Ylm_base_id + ", Abs[m]*2 + 1, "] = temp_value * ", imStr, ";"]
  ];
  Print[""];
];



(*运行生成 L=6*)
(* mList = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6};
Do[GenerateCppCode[6, m], {m, mList}] *)

(*运行生成 L=3*)
(* mList = {0, 1, -1, 2, -2, 3, -3};
Do[GenerateCppCode[3, m], {m, mList}] *)



(* =======================================================获得dxdydz================================================================= *)
(* !/usr/bin/env wolframscript *)

(* 1. 定义假设条件 *)
$Assumptions = {x \[Element] Reals, y \[Element] Reals, 
   z \[Element] Reals, rB > 0, thetaB > 0, thetaB < Pi, phiB > 0, 
   phiB < 2*Pi};

(* 2. 定义基础坐标变换关系 *)
rFunc = Sqrt[x^2 + y^2 + z^2];
thetaFunc = ArcCos[z / rFunc];
phiFunc = ArcTan[x, y]; (* 注意：Mathematica 中 ArcTan[x,y] 对应 C++ 的 atan2(y,x) *)

(* 3. 增强型格式化函数 (递归处理 POW 和变量名) *)
formatStr[expr_] := Module[{s, rules},
  (* N[..., 16] 确保输出小数，FullSimplify 确保提取公因式 *)
  s = ToString[CForm[N[FullSimplify[expr], 16]]];
  rules = {
    "rB" -> "r",
    "Power(" ~~ base : Shortest[__] ~~ "," ~~ exp : DigitCharacter .. ~~ ")" :> 
      "POW" <> exp <> "(" <> base <> ")",
    "Cos(thetaB)" -> "cos_theta", "Sin(thetaB)" -> "sin_theta",
    "Cos(phiB)" -> "cos_phi", "Sin(phiB)" -> "sin_phi",
    "Cos(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "thetaB)" :> "cos_" <> d <> "theta",
    "Sin(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "thetaB)" :> "sin_" <> d <> "theta",
    "Cos(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "phiB)" :> "cos_" <> d <> "phi",
    "Sin(" ~~ d : DigitCharacter .. ~~ (Blank[] | ".*") ~~ "phiB)" :> "sin_" <> d <> "phi",
    " " -> ""
  };
  FixedPoint[StringReplace[#, rules] &, s]
];

(* 定义一个更强力的公因式提取函数 *)
GetCommon[list_] := Module[{factors},
  (* 将列表转化为乘积形式，利用 Factor 自动提取 *)
  factors = Factor[Internal`CommonFactors[list]];
  Return[factors];
];


(* 4. 自动求导生成函数 *)
GenerateAutoDerivative[l_, m_] := Module[
  {ylm, gradX, gradY, gradZ, finalX, finalY, finalZ},
  (* 转换回角度表示，并提取公因式 1/r *)
  (* 关键步骤：通过替换将 x, y, z 转回 thetaB, phiB 形式以便输出代码 *)
  toAngles = {
    x -> rB * Sin[thetaB] * Cos[phiB],
    y -> rB * Sin[thetaB] * Sin[phiB],
    z -> rB * Cos[thetaB]
  };
  
  (* 直接定义球谐函数，并带入 thetaB 和 phiB 的函数定义 *)
  ylm = SphericalHarmonicY[l, m, thetaFunc, phiFunc];
  (* 让 Mathematica 自动处理复杂的链式求导 *)
  gradX = D[ylm, x];
  gradY = D[ylm, y];
  gradZ = D[ylm, z];

  finalX = FullSimplify[gradX /. toAngles];
  finalY = FullSimplify[gradY /. toAngles];
  finalZ = FullSimplify[gradZ /. toAngles];

  finalX = ComplexExpand[finalX];
  finalY = ComplexExpand[finalY];
  finalZ = ComplexExpand[finalZ];

(*  Print["// --- l=", l, ", m=", m, " Auto-Generated Derivatives ---"];

  (* 再次强调：这里手动乘以 rB 是为了在输出中提取出 (1.0/r) *)
  Print["dY", l, m, "_dx_re = (1.0/r) * (", formatStr[rB * Re[finalX]], ");"];
  Print["dY", l, m, "_dx_im = (1.0/r) * (", formatStr[rB * Im[finalX]], ");"];
  
  Print["dY", l, m, "_dy_re = (1.0/r) * (", formatStr[rB * Re[finalY]], ");"];
  Print["dY", l, m, "_dy_im = (1.0/r) * (", formatStr[rB * Im[finalY]], ");"];
  
  Print["dY", l, m, "_dz_re = (1.0/r) * (", formatStr[rB * Re[finalZ]], ");"];
  Print["dY", l, m, "_dz_im = (1.0/r) * (", formatStr[rB * Im[finalZ]], ");"];
  Print[""]; *)
  exprs = {
    Re[finalX],
    Im[finalX],
    Re[finalY],
    Im[finalY],
    Re[finalZ],
    Im[finalZ]
  };

  tags = {"dx", "dy", "dz"};
  dirGCDs = {};
  reducedCore = {};
  (* 1. 先计算各方向实部虚部之间的局部公项 *)
  Do[
      idx = (j - 1)*2 + 1;
      localGCD = PolynomialGCD[FullSimplify[exprs[[idx]]], FullSimplify[exprs[[idx + 1]]]];
      (* Print[localGCD]; *)
      AppendTo[dirGCDs, localGCD];
  , {j, 1, 3}];

  (* 2. 再提取三个局部公因式的公因式作为全局公项 *)
  globalGCD = PolynomialGCD[dirGCDs[[1]], dirGCDs[[2]], dirGCDs[[3]]];

  (* 3. 最终输出 *)
  Print["// --- l=", l, ", m=", m, " Hierarchical Derivatives ---"];
  Print["Factor_Y", " = (", formatStr[globalGCD], ");"];

  Do[
      idx = (j - 1)*2 + 1;
      (* 该方向最终输出的局部公因子（已除去全局项） *)
      finalDirGCD = Factor[dirGCDs[[j]] / globalGCD];
      
      (* Print["// Direction ", tags[[j]]]; *)
      Print["Factor_Y", tags[[j]], " = ", formatStr[finalDirGCD], ";"];
      
      (* 核心项：彻底除去所有公因子 *)
      tr = FullSimplify[exprs[[idx]] / (globalGCD * finalDirGCD)];
      ti = FullSimplify[exprs[[idx + 1]] / (globalGCD * finalDirGCD)];
      
      Print["t", tags[[j]], "_r = ", formatStr[tr], ";"];
      Print["t", tags[[j]], "_i = ", formatStr[ti], ";"];
  , {j, 1, 3}];
  Print[""];
];

(* 运行生成 L=6, m=0 示例 *)
(* mList = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6};
Do[GenerateAutoDerivative[6, m], {m, mList}] *)

(* 运行生成 L=4, m=0 示例 *)
(* mList = {0, 1, -1, 2, -2, 3, -3, 4, -4};
Do[GenerateAutoDerivative[4, m], {m, mList}] *)

(* 运行生成 L=3, m=0 示例 *)
mList = {0, 1, -1, 2, -2, 3, -3};
Do[GenerateAutoDerivative[3, m], {m, mList}]
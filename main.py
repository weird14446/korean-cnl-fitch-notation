from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union


# =====================================================================
# 1. 1차 논리 추상 구문 트리 (First-Order Logic AST)
# =====================================================================

class Term:
    def substitute(self, target: Term, replacement: Term) -> Term:
        raise NotImplementedError

    def free_vars(self) -> Set[str]:
        raise NotImplementedError

    def constants(self) -> Set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Var(Term):
    name: str

    def __str__(self) -> str:
        return self.name

    def substitute(self, target: Term, replacement: Term) -> Term:
        return replacement if self == target else self

    def free_vars(self) -> Set[str]:
        return {self.name}

    def constants(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class Const(Term):
    name: str

    def __str__(self) -> str:
        return self.name

    def substitute(self, target: Term, replacement: Term) -> Term:
        return replacement if self == target else self

    def free_vars(self) -> Set[str]:
        return set()

    def constants(self) -> Set[str]:
        return {self.name}


@dataclass(frozen=True)
class Func(Term):
    name: str
    args: Tuple[Term, ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.args))})"

    def substitute(self, target: Term, replacement: Term) -> Term:
        if self == target:
            return replacement
        return Func(self.name, tuple(a.substitute(target, replacement) for a in self.args))

    def free_vars(self) -> Set[str]:
        res: Set[str] = set()
        for a in self.args:
            res |= a.free_vars()
        return res

    def constants(self) -> Set[str]:
        res: Set[str] = set()
        for a in self.args:
            res |= a.constants()
        return res


class Formula:
    def substitute(self, target: Term, replacement: Term) -> Formula:
        raise NotImplementedError

    def free_vars(self) -> Set[str]:
        raise NotImplementedError

    def constants(self) -> Set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Bottom(Formula):
    def __str__(self) -> str:
        return "⊥"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return self

    def free_vars(self) -> Set[str]:
        return set()

    def constants(self) -> Set[str]:
        return set()


@dataclass(frozen=True)
class Pred(Formula):
    name: str
    args: Tuple[Term, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return Pred(self.name, tuple(a.substitute(target, replacement) for a in self.args))

    def free_vars(self) -> Set[str]:
        res: Set[str] = set()
        for a in self.args:
            res |= a.free_vars()
        return res

    def constants(self) -> Set[str]:
        res: Set[str] = set()
        for a in self.args:
            res |= a.constants()
        return res


@dataclass(frozen=True)
class Eq(Formula):
    left: Term
    right: Term

    def __str__(self) -> str:
        return f"({self.left} = {self.right})"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return Eq(
            self.left.substitute(target, replacement),
            self.right.substitute(target, replacement),
        )

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def constants(self) -> Set[str]:
        return self.left.constants() | self.right.constants()


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula

    def __str__(self) -> str:
        return f"¬{self.sub}"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return Not(self.sub.substitute(target, replacement))

    def free_vars(self) -> Set[str]:
        return self.sub.free_vars()

    def constants(self) -> Set[str]:
        return self.sub.constants()


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return And(
            self.left.substitute(target, replacement),
            self.right.substitute(target, replacement),
        )

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def constants(self) -> Set[str]:
        return self.left.constants() | self.right.constants()


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return Or(
            self.left.substitute(target, replacement),
            self.right.substitute(target, replacement),
        )

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def constants(self) -> Set[str]:
        return self.left.constants() | self.right.constants()


@dataclass(frozen=True)
class Implies(Formula):
    ant: Formula
    cons: Formula

    def __str__(self) -> str:
        return f"({self.ant} → {self.cons})"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return Implies(
            self.ant.substitute(target, replacement),
            self.cons.substitute(target, replacement),
        )

    def free_vars(self) -> Set[str]:
        return self.ant.free_vars() | self.cons.free_vars()

    def constants(self) -> Set[str]:
        return self.ant.constants() | self.cons.constants()


@dataclass(frozen=True)
class Iff(Formula):
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        return Iff(
            self.left.substitute(target, replacement),
            self.right.substitute(target, replacement),
        )

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def constants(self) -> Set[str]:
        return self.left.constants() | self.right.constants()


@dataclass(frozen=True)
class Forall(Formula):
    var: str
    body: Formula

    def __str__(self) -> str:
        return f"∀{self.var} {self.body}"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        if isinstance(target, Var) and target.name == self.var:
            return self
        if self.var in replacement.free_vars() and (
            isinstance(target, Var) and target.name in self.body.free_vars()
        ):
            raise ValueError(f"변수 포획(Variable Capture) 발생: '{replacement}'가 양화사 ∀{self.var}에 포획됨")
        return Forall(self.var, self.body.substitute(target, replacement))

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}

    def constants(self) -> Set[str]:
        return self.body.constants()


@dataclass(frozen=True)
class Exists(Formula):
    var: str
    body: Formula

    def __str__(self) -> str:
        return f"∃{self.var} {self.body}"

    def substitute(self, target: Term, replacement: Term) -> Formula:
        if isinstance(target, Var) and target.name == self.var:
            return self
        if self.var in replacement.free_vars() and (
            isinstance(target, Var) and target.name in self.body.free_vars()
        ):
            raise ValueError(f"변수 포획(Variable Capture) 발생: '{replacement}'가 양화사 ∃{self.var}에 포획됨")
        return Exists(self.var, self.body.substitute(target, replacement))

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}

    def constants(self) -> Set[str]:
        return self.body.constants()


def flatten_and(f: Formula) -> List[Formula]:
    """연언(∧) 결합법칙에 영향받지 않도록 평탄화된 연언지 리스트 반환"""
    if isinstance(f, And):
        return flatten_and(f.left) + flatten_and(f.right)
    return [f]


def flatten_or(f: Formula) -> List[Formula]:
    """선언(∨) 결합법칙에 영향받지 않도록 평탄화된 선언지 리스트 반환"""
    if isinstance(f, Or):
        return flatten_or(f.left) + flatten_or(f.right)
    return [f]


def ast_equiv(f1: Formula, f2: Formula) -> bool:
    """연언/선언의 결합 순서 차이를 허용하는 구문적 동치 비교"""
    if f1 == f2:
        return True
    if isinstance(f1, And) and isinstance(f2, And):
        c1, c2 = flatten_and(f1), flatten_and(f2)
        return len(c1) == len(c2) and all(ast_equiv(a, b) for a, b in zip(c1, c2))
    if isinstance(f1, Or) and isinstance(f2, Or):
        d1, d2 = flatten_or(f1), flatten_or(f2)
        return len(d1) == len(d2) and all(ast_equiv(a, b) for a, b in zip(d1, d2))
    if isinstance(f1, Implies) and isinstance(f2, Implies):
        return ast_equiv(f1.ant, f2.ant) and ast_equiv(f1.cons, f2.cons)
    if isinstance(f1, Iff) and isinstance(f2, Iff):
        return ast_equiv(f1.left, f2.left) and ast_equiv(f1.right, f2.right)
    if isinstance(f1, Not) and isinstance(f2, Not):
        return ast_equiv(f1.sub, f2.sub)
    if isinstance(f1, Forall) and isinstance(f2, Forall) and f1.var == f2.var:
        return ast_equiv(f1.body, f2.body)
    if isinstance(f1, Exists) and isinstance(f2, Exists) and f1.var == f2.var:
        return ast_equiv(f1.body, f2.body)
    return False


# =====================================================================
# 2. 한국어 통제 자연어(CNL v3.5) 파서 및 구문 정규화 엔진
# =====================================================================

@dataclass
class ProofStep:
    line_num: int
    depth: int
    is_assumption: bool
    flagged_const: Optional[str]
    rule_name: str
    refs: List[Union[int, Tuple[int, int]]]
    substs: List[Tuple[str, Term, Term]]  # ('assign'|'gen'|'replace', left, right)
    formula: Formula
    raw_text: str


class KoreanCNLParser:
    VAR_PATTERN = re.compile(r"^[xyzuvw][0-9]*$")
    PROP_PATTERN = re.compile(r"^[A-Z][0-9]*$")

    # 용언 어간/활용형 정규화 사전 (규칙 기반 + 불규칙 어간 매핑)
    VERB_STEM_MAP = {
        "죽는": "죽는다", "죽으": "죽는다", "죽": "죽는다",
        "크": "크다", "큰": "크다",
        "작": "작다", "작은": "작다",
        "작거나 같": "작거나같다", "작거나 같은": "작거나같다",
        "크거나 같": "크거나같다", "크거나 같은": "크거나같다",
        "사랑하는": "사랑한다", "사랑하": "사랑한다",
        "나누는": "나눈다", "나누": "나눈다",
        "속하는": "속한다", "속하": "속한다",
        "인접하는": "인접한다", "인접하": "인접한다",
    }

    def parse_term(self, s: str) -> Term:
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()

        # 기호 함수 호출: f(x, y)
        m_func = re.match(r"^([fgh][0-9]*)\((.+)\)$", s)
        if m_func:
            fname = m_func.group(1)
            args = tuple(self.parse_term(a) for a in m_func.group(2).split(","))
            return Func(fname, args)

        # 2항 자연어 함수: "x와 y의 합"
        m_bin = re.match(r"^(.+?)(?:과|와)\s+(.+?)의\s+([가-힣]+)$", s)
        if m_bin:
            t1 = self.parse_term(m_bin.group(1))
            t2 = self.parse_term(m_bin.group(2))
            return Func(m_bin.group(3), (t1, t2))

        # 1항 자연어 함수: "소크라테스의 스승", "a의 다음수"
        m_un = re.match(r"^(.+?)의\s+([가-힣]+)$", s)
        if m_un:
            return Func(m_un.group(2), (self.parse_term(m_un.group(1)),))

        if self.VAR_PATTERN.match(s):
            return Var(s)
        return Const(s)

    def normalize_verb(self, raw_verb: str) -> str:
        raw_verb = raw_verb.strip()
        if raw_verb in self.VERB_STEM_MAP:
            return self.VERB_STEM_MAP[raw_verb]
        if raw_verb.endswith("다"):
            return raw_verb
        return raw_verb + "다"

    def strip_quote_suffix(self, text: str) -> str:
        """인용절('~이라고', '~다고', '~라고')을 평서형 명제 텍스트로 복원"""
        text = text.strip()
        replacements = [
            ("같지 않다고", "같지 않다"),
            ("같다고", "같다"),
            ("같다라고", "같다"),
            ("다르다고", "다르다"),
            ("다르다라고", "다르다"),
            ("존재하지 않는다고", "존재하지 않는다"),
            ("존재한다고", "존재한다"),
            ("존재한다라고", "존재한다"),
            ("지 않는다고", "지 않는다"),
            ("지 않는다라고", "지 않는다"),
            ("는다라고", "는다"),
            ("는다고", "는다"),
            ("다라고", "다"),
            ("아니라고", "아니다"),
            ("아니다라고", "아니다"),
            ("이라고", "이다"),
            ("이라", "이다"),
            ("이다고", "이다"),
            ("이다라고", "이다"),
            ("라고", ""),
            ("다고", "다"),
            ("고", ""),
        ]
        for suf, rep in replacements:
            if text.endswith(suf):
                res = text[: -len(suf)] + rep
                # "인간"처럼 어미가 떨어진 명사형이나 "Q" 같은 명제상수 보정
                if suf == "라고" and not res.endswith("다"):
                    res = res + "이다"
                return res.strip()
        return text

    def extract_subject(self, text: str) -> Tuple[Optional[Union[Term, List[Term]]], str]:
        """문장 선두에서 단일 주어 또는 '갑과 을은 모두' 복수 주어를 분리"""
        text = text.strip()
        # 1) 복수 주어 분배 패턴: "갑과 을은 모두 ..."
        m_plural = re.match(r"^([^\s,]+?)(?:과|와)\s+([^\s,]+?)(?:은|는|이|가)\s+모두\s+(.+)$", text)
        if m_plural:
            t1 = self.parse_term(m_plural.group(1))
            t2 = self.parse_term(m_plural.group(2))
            return [t1, t2], m_plural.group(3).strip()

        # 2) 복합 항(함수) 주어 패턴: "a의 다음수는 ...", "x와 y의 합은 ..."
        m_complex_subj = re.match(
            r"^((?:[^\s,]+?(?:과|와)\s+)?[^\s,]+?의\s+[가-힣]+|[가-힣a-zA-Z0-9_]+)(?:은|는|이|가|도)\s+(.+)$",
            text,
        )
        if m_complex_subj:
            cand = m_complex_subj.group(1).strip()
            rest = m_complex_subj.group(2).strip()
            # '다음은 거짓이다' 등 예약어 제외
            if cand not in ("다음", "역"):
                return self.parse_term(cand), rest

        return None, text

    def parse_adnominal_modifier(self, mod_text: str, subj: Term) -> Formula:
        """관형절 수식어('철학자이고 아테네인인', '소크라테스의 제자이거나 소크라테스와 같은', '2보다 큰')를 Formula로 분해"""
        mod_text = mod_text.strip()

        # 선언형 관형절: "~이거나 ~인/같은"
        if "이거나 " in mod_text or "거나 " in mod_text:
            parts = re.split(r"(?:이거나|거나)\s+", mod_text, maxsplit=1)
            left_f = self.parse_vp(parts[0] + "이다", subj)
            right_f = self.parse_adnominal_modifier(parts[1], subj)
            return Or(left_f, right_f)

        # 연언형 관형절: "~이고 ~인/같은"
        if "이고 " in mod_text or "고 " in mod_text:
            parts = re.split(r"(?:이고|고)\s+", mod_text, maxsplit=1)
            left_s = parts[0] + ("이다" if mod_text.startswith(parts[0] + "이고") else "다")
            left_f = self.parse_vp(left_s, subj)
            right_f = self.parse_adnominal_modifier(parts[1], subj)
            return And(left_f, right_f)

        # 등치 관형절: "소크라테스와 같은"
        m_eq = re.match(r"^(.+?)(?:과|와)\s+같은$", mod_text)
        if m_eq:
            return Eq(subj, self.parse_term(m_eq.group(1)))

        # 비교/처소 관형절: "2보다 큰", "A에 속하는"
        m_comp = re.match(r"^(.+?)(보다|에|에서)\s+([가-힣]+)$", mod_text)
        if m_comp:
            obj_t = self.parse_term(m_comp.group(1))
            v_name = self.normalize_verb(m_comp.group(3))
            return Pred(v_name, (subj, obj_t))

        # 관계 명사 관형절: "A의 원소인", "소크라테스의 제자인"
        m_rel = re.match(r"^(.+?)의\s+([가-힣]+)인$", mod_text)
        if m_rel:
            return Pred(m_rel.group(2), (subj, self.parse_term(m_rel.group(1))))

        # 1항 명사 관형절: "철학자인", "짝수인"
        if mod_text.endswith("인"):
            return Pred(mod_text[:-1].strip(), (subj,))

        return self.parse_vp(mod_text + "이다", subj)

    def parse_vp(self, vp_text: str, subj: Term) -> Formula:
        """주어(subj)가 결정된 상태에서 서술구(VP)를 파싱 (관형절 수식 분해 포함)"""
        vp_text = vp_text.strip()

        # 1) 등치 및 부등치: "y와 같다", "y와 다르다", "y와 같지 않다"
        m_eq = re.match(r"^(.+?)(?:과|와)\s+(같다|같지 않다|다르다)$", vp_text)
        if m_eq:
            rhs = self.parse_term(m_eq.group(1))
            if m_eq.group(2) == "같다":
                return Eq(subj, rhs)
            return Not(Eq(subj, rhs))

        # 2) 관형절이 수식된 명사 술어: "철학자이고 아테네인인 인간이다", "2보다 큰 짝수인 소수이다"
        m_adn = re.match(r"^(.+?(?:인|큰|작은|같은))\s+([가-힣]+)(이다|(?:이|가)\s+아니다)$", vp_text)
        if m_adn:
            mod_str, head_noun, copula = m_adn.group(1), m_adn.group(2), m_adn.group(3)
            head_pred = Pred(head_noun, (subj,))
            if "아니다" in copula:
                head_pred = Not(head_pred)
            mod_formula = self.parse_adnominal_modifier(mod_str, subj)
            # 도메인 명사(head_pred)를 앞에 배치하여 제한된 양화사 정규화 순서(인간(x) ∧ ...)와 일치시킴
            return And(head_pred, mod_formula)

        # 3) 2항 관계 명사 술어: "y의 약수이다", "B의 원소가 아니다"
        m_rel = re.match(r"^(.+?)의\s+([가-힣]+?)(이다|(?:이|가)\s+아니다)$", vp_text)
        if m_rel:
            arg2 = self.parse_term(m_rel.group(1))
            pred = Pred(m_rel.group(2), (subj, arg2))
            return Not(pred) if "아니다" in m_rel.group(3) else pred

        # 4) 1항 명사 술어: "인간이다", "인간이 아니다"
        m_nom = re.match(r"^([가-힣A-Z0-9_]+?)(이다|(?:이|가)\s+아니다)$", vp_text)
        if m_nom:
            noun = m_nom.group(1)
            if self.PROP_PATTERN.match(noun):
                p = Pred(noun, ())
            else:
                p = Pred(noun, (subj,))
            return Not(p) if "아니다" in m_nom.group(2) else p

        # 5) 2항 비교/대칭/타동 용언 술어: "y보다 작거나 같다", "y를 사랑하지 않는다"
        m_bin_v = re.match(r"^(.+?)(보다|과|와|을|를|에|에서)\s+(.+)$", vp_text)
        if m_bin_v:
            arg2 = self.parse_term(m_bin_v.group(1))
            v_part = m_bin_v.group(3).strip()
            if v_part.endswith("지 않는다"):
                stem = v_part[: -len("지 않는다")].strip()
                return Not(Pred(self.normalize_verb(stem), (subj, arg2)))
            return Pred(self.normalize_verb(v_part), (subj, arg2))

        # 6) 1항 용언 술어 (긍정/부정): "죽는다", "죽지 않는다"
        if vp_text.endswith("지 않는다"):
            stem = vp_text[: -len("지 않는다")].strip()
            return Not(Pred(self.normalize_verb(stem), (subj,)))

        return Pred(self.normalize_verb(vp_text), (subj,))

    def normalize_connective_clause(self, clause: str) -> str:
        """연결형/조건형 어미('~이고', '~이거나', '~이면')로 끝나는 절을 평서형('~이다/다')으로 복원"""
        clause = clause.strip().rstrip(",")
        rules = [
            ("같지 않으면", "같지 않다"), ("같지 않고", "같지 않다"), ("같지 않거나", "같지 않다"),
            ("같으면", "같다"), ("같고", "같다"), ("같거나", "같다"),
            ("다르면", "다르다"), ("다르고", "다르다"), ("다르거나", "다르다"),
            ("지 않으면", "지 않는다"), ("지 않고", "지 않는다"), ("지 않거나", "지 않는다"),
            ("아니면", "아니다"), ("아니고", "아니다"), ("아니거나", "아니다"),
            ("이라면", "이다"), ("이면", "이다"), ("이고", "이다"), ("이며", "이다"), ("이거나", "이다"),
            ("으면", "는다"), ("면", "다"), ("고", "다"), ("거나", "다"),
        ]
        for suf, rep in rules:
            if clause.endswith(suf):
                stem = clause[: -len(suf)]
                if suf == "고" and stem in self.VERB_STEM_MAP:
                    return self.VERB_STEM_MAP[stem]
                return stem + rep
        return clause

    def parse_prop(self, text: str, inherited_subj: Optional[Term] = None) -> Tuple[Formula, Optional[Term]]:
        """
        한국어 CNL 명제를 표준 1차 논리 AST로 파싱하며,
        현재 절의 주어를 반환하여 후행 절의 주어 생략 복원(Ellipsis Resolution)에 전달함.
        """
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            # 최외곽 괄호가 전체를 감싸는 경우 제거
            depth = 0
            balanced = True
            for i, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and i < len(text) - 1:
                    balanced = False
                    break
            if balanced:
                text = text[1:-1].strip()

        # 0. 모순 및 순수 명제 상수
        if text in ("모순이 발생한다", "모순이다", "모순"):
            return Bottom(), inherited_subj
        if self.PROP_PATTERN.match(text) or (
            text.endswith("이다") and self.PROP_PATTERN.match(text[:-2])
        ):
            pname = text[:-2] if text.endswith("이다") else text
            return Pred(pname, ()), inherited_subj
        m_prop_neg = re.match(r"^([A-Z][0-9]*)(?:이|가)\s+아니다$", text)
        if m_prop_neg:
            return Not(Pred(m_prop_neg.group(1), ())), inherited_subj

        # 1. 전위 부정 ("다음은 거짓이다: ...", "아니다: ...")
        for neg_prefix in ("다음은 거짓이다:", "아니다:"):
            if text.startswith(neg_prefix):
                sub_f, subj = self.parse_prop(text[len(neg_prefix) :].strip(), inherited_subj)
                return Not(sub_f), subj

        # 2. 후위 관형절형 존재양화 ("~인 [도메인] x가 존재한다 / 존재하지 않는다")
        m_post_ex = re.match(
            r"^(.+?)\s+([xyzuvw][0-9]*)(?:이|가|은|는)\s+(존재한다|존재하지 않는다)$",
            text,
        )
        if m_post_ex and not text.startswith(("모든 ", "임의의 ", "어떤 ", "만약 ")):
            desc_part = m_post_ex.group(1).strip()
            var_name = m_post_ex.group(2).strip()
            is_neg = m_post_ex.group(3) == "존재하지 않는다"
            v_term = Var(var_name)

            # 도메인 명사가 분리되어 있는지 확인 (예: "철학자이고 아테네인인 인간")
            m_dom = re.match(r"^(.+?(?:인|큰|작은|같은))\s+([가-힣]+)$", desc_part)
            if m_dom:
                mod_f = self.parse_adnominal_modifier(m_dom.group(1), v_term)
                dom_f = Pred(m_dom.group(2), (v_term,))
                body_f: Formula = And(dom_f, mod_f)
            else:
                body_f = self.parse_adnominal_modifier(desc_part, v_term)

            ex_ast: Formula = Exists(var_name, body_f)
            return (Not(ex_ast) if is_neg else ex_ast), None

        # 3. 전위 양화사 (제한된 보편양화 / 관계 영역 양화 / 다중 변수 양화)
        # 3-a) 관계 명사 영역 양화: "A의 모든 원소 x에 대해, ..."
        m_rel_quant = re.match(
            r"^(.+?)의\s+(모든|임의의|어떤)\s+([가-힣]+?)(?:인)?\s+([xyzuvw0-9,\s]+)에\s+대해,\s*(.+)$",
            text,
        )
        if m_rel_quant:
            rel_arg = self.parse_term(m_rel_quant.group(1))
            q_type = m_rel_quant.group(2)
            rel_noun = m_rel_quant.group(3)
            vars_list = [v.strip() for v in m_rel_quant.group(4).split(",")]
            body_f, _ = self.parse_prop(m_rel_quant.group(5))

            for v in reversed(vars_list):
                dom_pred = Pred(rel_noun, (Var(v), rel_arg))
                if q_type in ("모든", "임의의"):
                    body_f = Forall(v, Implies(dom_pred, body_f))
                else:
                    body_f = Exists(v, And(dom_pred, body_f))
            return body_f, None

        # 3-b) 일반 및 1항 도메인 제한 양화: "모든 [인간] x, y에 대해, ..."
        m_quant = re.match(
            r"^(모든|임의의|어떤)\s+(?:([가-힣]+?)(?:인|의)?\s+)?([xyzuvw0-9,\s]+?)(?:에\s+대해,|이\s+존재하여,|가\s+존재하여,)\s*(.+)$",
            text,
        )
        if m_quant:
            q_type = m_quant.group(1)
            domain_noun = m_quant.group(2)
            vars_list = [v.strip() for v in m_quant.group(3).split(",")]
            body_f, _ = self.parse_prop(m_quant.group(4))

            for v in reversed(vars_list):
                if domain_noun:
                    dom_pred = Pred(domain_noun, (Var(v),))
                    if q_type in ("모든", "임의의"):
                        body_f = Forall(v, Implies(dom_pred, body_f))
                    else:
                        body_f = Exists(v, And(dom_pred, body_f))
                else:
                    body_f = Forall(v, body_f) if q_type in ("모든", "임의의") else Exists(v, body_f)
            return body_f, None

        # 4. 조건문 ("만약 ~이면, ~" 또는 "~일 때, ~") - 전건의 주어를 후건으로 상속
        m_outer_when = re.match(r"^(.+?)일\s+때,\s*(.+)$", text)
        if m_outer_when and not text.startswith("만약 "):
            ant_str = m_outer_when.group(1).strip() + "이다"
            cons_str = m_outer_when.group(2).strip()
            ant_f, ant_subj = self.parse_prop(ant_str, inherited_subj)
            cons_f, _ = self.parse_prop(cons_str, ant_subj or inherited_subj)
            return Implies(ant_f, cons_f), ant_subj

        if text.startswith("만약 "):
            rest = text[len("만약 ") :].strip()
            m_cond = re.match(r"^(.+?(?:이면|라면|한다면|면)),\s*(.+)$", rest)
            if m_cond:
                ant_raw = m_cond.group(1).strip()
                cons_raw = m_cond.group(2).strip()
                ant_f, ant_subj = self.parse_condition_antecedent(ant_raw, inherited_subj)
                cons_f, _ = self.parse_prop(cons_raw, ant_subj or inherited_subj)
                return Implies(ant_f, cons_f), ant_subj

        # 5. 선언문 (", 또는" / "~이거나, " / "~이거나 ") - 좌측 주어를 우측 무주어 절에 상속
        m_or = re.split(r",\s*또는\s+|(?<=이거나),\s+|(?<=거나),\s+|(?<=이거나)\s+|(?<=거나)\s+", text, maxsplit=1)
        if len(m_or) == 2:
            # 분리된 지점 앞 어미를 복원하기 위해 원본에서의 위치 계산
            sep_match = re.search(r",\s*또는\s+|이거나,\s+|거나,\s+|이거나\s+|거나\s+", text)
            assert sep_match is not None
            left_raw = text[: sep_match.end()].strip()
            right_raw = text[sep_match.end() :].strip()
            if left_raw.endswith("또는"):
                left_raw = text[: sep_match.start()].strip()
            left_norm = self.normalize_connective_clause(left_raw)
            left_f, left_subj = self.parse_prop(left_norm, inherited_subj)
            right_f, _ = self.parse_prop(right_raw, left_subj or inherited_subj)
            return Or(left_f, right_f), left_subj or inherited_subj

        # 6. 연언문 (", 그리고" / "~이고, " / "~이고 ") - 좌측 주어를 우측 무주어 절에 상속
        sep_and = re.search(r",\s*그리고\s+|이고,\s+|이며,\s+|고,\s+|이고\s+|고\s+", text)
        if sep_and:
            # 단, 관형절 내부("철학자이고 아테네인인 인간이다")인 경우는 원자 명제 단계에서 처리하도록 예외 처리
            right_preview = text[sep_and.end() :].strip()
            if not re.match(r"^[가-힣]+인\s+[가-힣]+(?:이다|이\s+아니다|가\s+아니다)$", right_preview):
                left_raw = text[: sep_and.end()].strip()
                if left_raw.endswith("그리고"):
                    left_raw = text[: sep_and.start()].strip()
                left_norm = self.normalize_connective_clause(left_raw)
                left_f, left_subj = self.parse_prop(left_norm, inherited_subj)
                right_f, _ = self.parse_prop(right_preview, left_subj or inherited_subj)
                return And(left_f, right_f), left_subj or inherited_subj

        # 7. 원자 명제 및 복수 주어('모두') 처리
        subj_extracted, vp_rest = self.extract_subject(text)
        if isinstance(subj_extracted, list):
            # "갑과 을은 모두 철학자이다" -> 철학자(갑) ∧ 철학자(을)
            t1, t2 = subj_extracted[0], subj_extracted[1]
            return And(self.parse_vp(vp_rest, t1), self.parse_vp(vp_rest, t2)), None

        active_subj = subj_extracted if isinstance(subj_extracted, Term) else inherited_subj
        if active_subj is None:
            raise ValueError(f"주어를 찾을 수 없으며 상속할 선행 주어도 없습니다: '{text}'")

        return self.parse_vp(vp_rest, active_subj), active_subj

    def parse_condition_antecedent(
        self, ant_raw: str, inherited_subj: Optional[Term]
    ) -> Tuple[Formula, Optional[Term]]:
        """조건문의 전건('x가 철학자이고 아테네인이면' 등)을 파싱하고 주어를 추출"""
        sep_and = re.search(r",\s*그리고\s+|이고,\s+|고,\s+|이고\s+|고\s+", ant_raw)
        if sep_and:
            left_raw = ant_raw[: sep_and.end()].strip()
            if left_raw.endswith("그리고"):
                left_raw = ant_raw[: sep_and.start()].strip()
            right_raw = ant_raw[sep_and.end() :].strip()
            left_norm = self.normalize_connective_clause(left_raw)
            left_f, left_subj = self.parse_prop(left_norm, inherited_subj)
            right_f, _ = self.parse_condition_antecedent(right_raw, left_subj or inherited_subj)
            return And(left_f, right_f), left_subj or inherited_subj

        norm = self.normalize_connective_clause(ant_raw)
        return self.parse_prop(norm, inherited_subj)

    def parse_subst_list(self, subst_str: str) -> List[Tuple[str, Term, Term]]:
        res: List[Tuple[str, Term, Term]] = []
        for item in subst_str.split(","):
            item = item.strip()
            if ":=" in item:
                k, v = item.split(":=", 1)
                res.append(("assign", self.parse_term(k), self.parse_term(v)))
            elif "->" in item:
                k, v = item.split("->", 1)
                res.append(("gen", self.parse_term(k), self.parse_term(v)))
            elif "=>" in item:
                k, v = item.split("=>", 1)
                res.append(("replace", self.parse_term(k), self.parse_term(v)))
        return res

    def parse_line(self, raw_line: str) -> ProofStep:
        line = raw_line.strip()
        m_head = re.match(r"^(\d+)\.\s*((?:\|\s*)*)(.+)$", line)
        if not m_head:
            raise ValueError(f"줄 번호 형식 오류: {raw_line}")

        line_num = int(m_head.group(1))
        bars_str = m_head.group(2)
        depth = bars_str.count("|")
        stmt = m_head.group(3).strip()

        # 1) 가정 및 전제 문장 판별 ("~ 가정하자.")
        if stmt.endswith("가정하자.") or stmt.endswith("도입하자."):
            flagged_const: Optional[str] = None
            body_stmt = stmt

            m_flag_assume = re.match(
                r"^(?:임의의\s+)?(?:새\s+)?상수\s+([^\s,]+?)에\s+대해,\s*(.+)$", body_stmt
            )
            if m_flag_assume:
                flagged_const = m_flag_assume.group(1).strip()
                body_stmt = m_flag_assume.group(2).strip()

            body_stmt = re.sub(r"^(?:전제로|가정으로)\s+", "", body_stmt)
            body_stmt = re.sub(r"\s*가정하자\.$", "", body_stmt)
            prop_str = self.strip_quote_suffix(body_stmt)
            formula, _ = self.parse_prop(prop_str)
            rule_label = "전제" if depth == 0 else (f"가정(새 상수: {flagged_const})" if flagged_const else "가정")
            return ProofStep(
                line_num=line_num,
                depth=depth,
                is_assumption=True,
                flagged_const=flagged_const,
                rule_name=rule_label,
                refs=[],
                substs=[],
                formula=formula,
                raw_text=raw_line.strip(),
            )

        # 2) 추론 도출 문장 판별 ("명제 1, 2로부터 규칙 ~에 의해, (따라서)? ~.")
        m_deriv = re.match(
            r"^(?:명제\s+([0-9,\-\s]+?)(?:으로부터|로부터|에서|에|과|와)\s+)?"
            r"(?:규칙|추론규칙)?\s*([가-힣]+)(?:\[(.+?)\])?(?:에\s+의해|(?:을|를)\s+적용하여),\s*"
            r"(?:(?:따라서|그러므로|즉|결국)\s+)?(.+)\.$",
            stmt,
        )
        if not m_deriv:
            raise ValueError(f"도출 문장 구문 분석 실패: {raw_line}")

        refs_str, rule_name, subst_str, prop_str = m_deriv.groups()
        refs: List[Union[int, Tuple[int, int]]] = []
        if refs_str:
            for r in refs_str.split(","):
                r = r.strip()
                if "-" in r:
                    s_idx, e_idx = r.split("-", 1)
                    refs.append((int(s_idx.strip()), int(e_idx.strip())))
                else:
                    refs.append(int(r))

        substs = self.parse_subst_list(subst_str) if subst_str else []
        formula, _ = self.parse_prop(prop_str.strip())

        return ProofStep(
            line_num=line_num,
            depth=depth,
            is_assumption=False,
            flagged_const=None,
            rule_name=rule_name.strip(),
            refs=refs,
            substs=substs,
            formula=formula,
            raw_text=raw_line.strip(),
        )


# =====================================================================
# 3. Fitch 스코프 및 자연연역 추론 규칙 검증기 (Fitch Verifier)
# =====================================================================

@dataclass
class SubproofBlock:
    start_line: int
    end_line: Optional[int]
    depth: int
    assumption: Formula
    flagged_const: Optional[str]
    conclusion: Optional[Formula] = None


class FitchVerifier:
    def __init__(self) -> None:
        self.parser = KoreanCNLParser()

    def verify_proof(self, proof_text: str) -> Tuple[bool, List[str]]:
        lines = [
            ln for ln in proof_text.strip().splitlines()
            if ln.strip() and not ln.strip().startswith("//")
        ]
        steps: dict[int, ProofStep] = {}
        open_blocks: List[SubproofBlock] = []
        closed_blocks: dict[Tuple[int, int], SubproofBlock] = {}
        logs: List[str] = []

        prev_depth = 0
        prev_line_num = 0

        for raw_line in lines:
            try:
                step = self.parser.parse_line(raw_line)
            except Exception as e:
                logs.append(f"[구문 오류] {raw_line.strip()} -> {e}")
                return False, logs

            # 1) 줄 번호 및 들여쓰기 깊이(Subproof Scope) 상태 전이 검사
            if step.line_num != prev_line_num + 1:
                logs.append(f"[줄 번호 오류] {step.line_num}번 줄: 순차적 번호가 아닙니다.")
                return False, logs

            if step.depth > prev_depth:
                if step.depth != prev_depth + 1 or not step.is_assumption:
                    logs.append(
                        f"[스코프 오류] {step.line_num}번 줄: 새 부분증명 블록은 반드시 '가정'으로 시작하며 깊이가 1씩 증가해야 합니다."
                    )
                    return False, logs
                # 고유상수(Eigenvalue) 중복 등장 검사
                if step.flagged_const:
                    used_consts: Set[str] = set()
                    for prev_s in steps.values():
                        if self._is_accessible(prev_s.line_num, open_blocks, closed_blocks):
                            used_consts |= prev_s.formula.constants()
                    if step.flagged_const in used_consts:
                        logs.append(
                            f"[고유상수 위반] {step.line_num}번 줄: 새 상수 '{step.flagged_const}'가 이미 외부 전제/가정에 등장했습니다."
                        )
                        return False, logs

                open_blocks.append(
                    SubproofBlock(
                        start_line=step.line_num,
                        end_line=None,
                        depth=step.depth,
                        assumption=step.formula,
                        flagged_const=step.flagged_const,
                    )
                )
            elif step.depth < prev_depth:
                while open_blocks and open_blocks[-1].depth > step.depth:
                    blk = open_blocks.pop()
                    blk.end_line = prev_line_num
                    blk.conclusion = steps[prev_line_num].formula
                    closed_blocks[(blk.start_line, blk.end_line)] = blk

                if step.is_assumption and step.depth > 0:
                    # 같은 깊이에서 연달아 새 서브프루프를 여는 경우
                    open_blocks.append(
                        SubproofBlock(
                            start_line=step.line_num,
                            end_line=None,
                            depth=step.depth,
                            assumption=step.formula,
                            flagged_const=step.flagged_const,
                        )
                    )

            # 2) 추론 규칙 타당성 검증
            ok, reason = self._check_rule(step, steps, open_blocks, closed_blocks)
            steps[step.line_num] = step
            prev_depth = step.depth
            prev_line_num = step.line_num

            bar_prefix = "| " * step.depth
            status_mark = "통과" if ok else "실패"
            logs.append(
                f"[{status_mark}] {step.line_num:2d}번 줄: {bar_prefix}{step.formula}  "
                f"<-- [{step.rule_name}] ({reason})"
            )
            if not ok:
                return False, logs

        return True, logs

    def _is_accessible(
        self,
        target_line: int,
        open_blocks: List[SubproofBlock],
        closed_blocks: dict[Tuple[int, int], SubproofBlock],
    ) -> bool:
        """참조된 줄이 이미 닫힌 부분증명 내부에 속해 있지 않은지 확인"""
        open_starts = {b.start_line for b in open_blocks}
        for (s_idx, e_idx), blk in closed_blocks.items():
            if s_idx <= target_line <= e_idx and s_idx not in open_starts:
                return False
        return True

    def _check_rule(
        self,
        step: ProofStep,
        steps: dict[int, ProofStep],
        open_blocks: List[SubproofBlock],
        closed_blocks: dict[Tuple[int, int], SubproofBlock],
    ) -> Tuple[bool, str]:
        if step.is_assumption:
            return True, "유효한 전제/가정 선언"

        # 참조된 단일 줄 및 부분증명 블록의 접근 가능성 확인
        line_refs: List[ProofStep] = []
        block_refs: List[SubproofBlock] = []
        for r in step.refs:
            if isinstance(r, int):
                if r not in steps:
                    return False, f"존재하지 않는 명제 {r} 참조"
                if not self._is_accessible(r, open_blocks, closed_blocks):
                    return False, f"명제 {r}번은 이미 닫힌 부분증명 내부에 있어 참조할 수 없음(Fitch 스코프 위반)"
                line_refs.append(steps[r])
            else:
                if r not in closed_blocks:
                    return False, f"부분증명 범위 {r[0]}-{r[1]}이(가) 올바르게 닫히지 않았거나 존재하지 않음"
                block_refs.append(closed_blocks[r])

        rule = step.rule_name
        goal = step.formula

        # --- 명제 논리 추론 규칙 ---
        if rule == "반복":
            if len(line_refs) == 1 and ast_equiv(line_refs[0].formula, goal):
                return True, f"명제 {line_refs[0].line_num} 동일 반복"
            return False, "참조 명제와 일치하지 않음"

        if rule == "연언도입":
            if len(line_refs) >= 2:
                combined: List[Formula] = []
                for lr in line_refs:
                    combined.extend(flatten_and(lr.formula))
                goal_conj = flatten_and(goal)
                if all(any(ast_equiv(g, c) for c in combined) for g in goal_conj):
                    return True, "연언지 결합 확인"
            return False, "연언도입 전제 불일치"

        if rule == "연언제거":
            if len(line_refs) == 1:
                avail = flatten_and(line_refs[0].formula)
                goal_conj = flatten_and(goal)
                if all(any(ast_equiv(g, a) for a in avail) for g in goal_conj):
                    return True, f"명제 {line_refs[0].line_num}의 연언지로부터 추출 일치"
            return False, "참조 명제의 연언지에서 목표 명제를 찾을 수 없음"

        if rule == "선언도입":
            if len(line_refs) == 1 and isinstance(goal, Or):
                disjuncts = flatten_or(goal)
                if any(ast_equiv(line_refs[0].formula, d) for d in disjuncts):
                    return True, "선언지 포함 확인"
            return False, "선언도입 불일치"

        if rule == "조건제거":
            if len(line_refs) == 2:
                f1, f2 = line_refs[0].formula, line_refs[1].formula
                for imp_cand, ant_cand in [(f1, f2), (f2, f1)]:
                    if isinstance(imp_cand, Implies) and ast_equiv(imp_cand.ant, ant_cand):
                        if ast_equiv(imp_cand.cons, goal):
                            return True, "조건문 전건 일치 및 후건 도출 확인"
            return False, "조건제거(Modus Ponens) 전건/후건 매칭 실패"

        if rule == "조건도입":
            if len(block_refs) == 1 and isinstance(goal, Implies):
                blk = block_refs[0]
                if blk.conclusion and ast_equiv(blk.assumption, goal.ant) and ast_equiv(blk.conclusion, goal.cons):
                    return True, f"부분증명 {blk.start_line}-{blk.end_line} 가정·결론 일치"
            return False, "조건도입 부분증명 매칭 실패"

        if rule == "모순도입":
            if len(line_refs) == 2 and isinstance(goal, Bottom):
                f1, f2 = line_refs[0].formula, line_refs[1].formula
                if (isinstance(f1, Not) and ast_equiv(f1.sub, f2)) or (
                    isinstance(f2, Not) and ast_equiv(f2.sub, f1)
                ):
                    return True, "명제와 그 부정으로부터 모순(⊥) 도출 확인"
            return False, "모순도입을 위한 서로 부정 관계인 두 명제가 아님"

        if rule in ("부정도입", "귀류법"):
            if len(block_refs) == 1 and block_refs[0].conclusion == Bottom():
                blk = block_refs[0]
                if isinstance(goal, Not) and ast_equiv(blk.assumption, goal.sub):
                    return True, f"부분증명 {blk.start_line}-{blk.end_line}에서 모순 도출로 부정 증명 완료"
            return False, "부정도입 부분증명의 결론이 모순(⊥)이 아니거나 가정이 일치하지 않음"

        if rule == "이중부정제거":
            if len(line_refs) == 1:
                f = line_refs[0].formula
                if isinstance(f, Not) and isinstance(f.sub, Not) and ast_equiv(f.sub.sub, goal):
                    return True, "이중부정 제거 확인"
            return False, "이중부정 형태가 아님"

        # --- 1차 술어 논리 추론 규칙 ---
        if rule == "보편제거":
            if len(line_refs) == 1 and step.substs:
                curr = line_refs[0].formula
                for stype, var_t, rep_t in step.substs:
                    if stype != "assign" or not isinstance(curr, Forall) or curr.var != str(var_t):
                        return False, f"보편양화 변수 {var_t} 매칭 실패"
                    curr = curr.body.substitute(var_t, rep_t)
                if ast_equiv(curr, goal):
                    return True, f"보편양화 변수 대입 일치"
            return False, "보편제거 대입 결과 불일치"

        if rule == "보편도입":
            if len(step.substs) == 1 and step.substs[0][0] == "gen":
                _, const_t, var_t = step.substs[0]
                if not isinstance(goal, Forall) or goal.var != str(var_t):
                    return False, "목표 명제가 해당 변수의 보편양화식이 아님"

                # 블록 인용 방식 보편도입 (예: 명제 3-8로부터 규칙 보편도입[c -> x])
                if len(block_refs) == 1:
                    blk = block_refs[0]
                    if blk.flagged_const and blk.flagged_const != str(const_t):
                        return False, f"부분증명의 고유상수({blk.flagged_const})와 일반화 상수({const_t}) 불일치"
                    # 1) 부분증명 마지막 줄이 조건문이고 그 안에서 c -> x 일반화하는 경우
                    if blk.conclusion and ast_equiv(blk.conclusion.substitute(const_t, var_t), goal.body):
                        if str(const_t) in goal.constants():
                            return False, f"고유상수 '{const_t}'가 결론에 아직 남아 있음"
                        return True, f"고유상수 '{const_t}'를 변수 '{var_t}'로 보편 일반화 완료"
                    # 2) 부분증명 가정이 도메인 D(c)이고 결론이 P(c)일 때 직접 제한된 보편양화로 닫는 경우
                    if blk.conclusion:
                        imp_formed = Implies(blk.assumption, blk.conclusion)
                        if ast_equiv(imp_formed.substitute(const_t, var_t), goal.body):
                            return True, f"부분증명 {blk.start_line}-{blk.end_line}을 제한된 보편양화로 일반화 완료"

                # 단일 줄 인용 방식 보편도입
                if len(line_refs) == 1:
                    for b in open_blocks:
                        if str(const_t) in b.assumption.constants():
                            return False, f"상수 '{const_t}'가 현재 열려 있는 가정에 포함되어 있어 일반화 불가"
                    if ast_equiv(line_refs[0].formula.substitute(const_t, var_t), goal.body):
                        return True, f"임의의 상수 '{const_t}'를 '{var_t}'로 일반화 완료"
            return False, "보편도입 조건 불일치"

        if rule == "존재도입":
            if len(line_refs) == 1 and len(step.substs) == 1 and isinstance(goal, Exists):
                _, src_term, var_t = step.substs[0]
                if goal.var == str(var_t):
                    # 목표 존재양화 몸체에서 x에 다시 src_term을 대입했을 때 전제 줄과 일치하는지 확인
                    instantiated = goal.body.substitute(var_t, src_term)
                    if ast_equiv(instantiated, line_refs[0].formula):
                        return True, f"항 '{src_term}'을 변수 '{var_t}'로 존재 일반화 확인"
            return False, "존재도입 치환 매칭 실패"

        if rule == "존재제거":
            if len(line_refs) == 1 and len(block_refs) == 1:
                ex_f = line_refs[0].formula
                blk = block_refs[0]
                if isinstance(ex_f, Exists) and blk.flagged_const and blk.conclusion:
                    c_term = Const(blk.flagged_const)
                    expected_assum = ex_f.body.substitute(Var(ex_f.var), c_term)
                    if not ast_equiv(blk.assumption, expected_assum):
                        return False, f"부분증명 가정이 존재양화 인스턴스 {expected_assum}와 일치하지 않음"
                    if not ast_equiv(blk.conclusion, goal):
                        return False, "부분증명 결론과 존재제거 도출 명제 불일치"
                    if blk.flagged_const in goal.constants():
                        return False, f"고유상수 누출 오류: 임시 상수 '{blk.flagged_const}'가 최종 결론에 포함될 수 없음"
                    return True, f"존재명제 {line_refs[0].line_num} 및 부분증명 {blk.start_line}-{blk.end_line} 검증 완료 (상수 '{blk.flagged_const}' 소거됨)"
            return False, "존재제거 인자(존재명제 1개 + 부분증명 블록 1개) 또는 조건 불일치"

        if rule == "등치도입":
            if isinstance(goal, Eq) and goal.left == goal.right:
                return True, "반사성(t = t) 확인"
            return False, "좌변과 우변이 동일한 항이 아님"

        if rule == "등치제거":
            if len(line_refs) == 2 and len(step.substs) == 1 and step.substs[0][0] == "replace":
                _, src_t, dst_t = step.substs[0]
                f1, f2 = line_refs[0].formula, line_refs[1].formula
                for eq_cand, target_f in [(f1, f2), (f2, f1)]:
                    if isinstance(eq_cand, Eq) and (
                        (eq_cand.left == src_t and eq_cand.right == dst_t)
                        or (eq_cand.right == src_t and eq_cand.left == dst_t)
                    ):
                        if ast_equiv(target_f.substitute(src_t, dst_t), goal):
                            return True, f"등치식 ({src_t} = {dst_t}) 치환 일치"
            return False, "등치제거 치환 결과 불일치"

        return False, f"지원되지 않거나 인자가 잘못된 규칙: {rule}"


# =====================================================================
# 4. 실행 및 증명 검증 테스트 스위트
# =====================================================================

if __name__ == "__main__":
    verifier = FitchVerifier()

    # [테스트 1] 한국어 자연화 문법 종합 증명
    # - 동일 주어 생략 복원 ("만약 x가 철학자이고 아테네인이면, 소크라테스의 제자이거나 소크라테스와 같다")
    # - 후위 존재양화 ("철학자이고 아테네인인 인간 x가 존재한다")
    # - 관형절 수식 분해 ("c는 철학자이고 아테네인인 인간이라고 가정하자")
    proof_1 = """
    1. 모든 인간 x에 대해, 만약 x가 철학자이고 아테네인이면, 소크라테스의 제자이거나 소크라테스와 같다고 가정하자.
    2. 철학자이고 아테네인인 인간 x가 존재한다고 가정하자.
    3. | 새 상수 c에 대해, c는 철학자이고 아테네인인 인간이라고 가정하자.
    4. | 명제 3으로부터 규칙 연언제거에 의해, c는 인간이다.
    5. | 명제 3으로부터 규칙 연언제거에 의해, c는 철학자이고 아테네인이다.
    6. | 명제 1로부터 규칙 보편제거[x := c]에 의해, c가 인간일 때, 만약 c가 철학자이고 아테네인이면, 소크라테스의 제자이거나 소크라테스와 같다.
    7. | 명제 4, 6으로부터 규칙 조건제거에 의해, 만약 c가 철학자이고 아테네인이면, 소크라테스의 제자이거나 소크라테스와 같다.
    8. | 명제 5, 7로부터 규칙 조건제거에 의해, c는 소크라테스의 제자이거나 소크라테스와 같다.
    9. | 명제 8로부터 규칙 존재도입[c -> x]에 의해, 소크라테스의 제자이거나 소크라테스와 같은 x가 존재한다.
    10. 명제 2, 3-9로부터 규칙 존재제거에 의해, 따라서 소크라테스의 제자이거나 소크라테스와 같은 x가 존재한다.
    """

    # [테스트 2] 관계 명사 영역 제한 양화('A의 모든 원소 x') 및 보편도입(∀I) 증명
    proof_2 = """
    1. A의 모든 원소 x에 대해, x는 B의 원소이라고 가정하자.
    2. B의 모든 원소 y에 대해, y는 C의 원소이라고 가정하자.
    3. | 임의의 새 상수 c에 대해, c는 A의 원소이라고 가정하자.
    4. | 명제 1로부터 규칙 보편제거[x := c]에 의해, 만약 c가 A의 원소이면, c는 B의 원소이다.
    5. | 명제 3, 4에 규칙 조건제거를 적용하여, c는 B의 원소이다.
    6. | 명제 2로부터 규칙 보편제거[y := c]에 의해, 만약 c가 B의 원소이면, c는 C의 원소이다.
    7. | 명제 5, 6에 규칙 조건제거를 적용하여, c는 C의 원소이다.
    8. 명제 3-7로부터 규칙 조건도입에 의해, 만약 c가 A의 원소이면, c는 C의 원소이다.
    9. 명제 3-8로부터 규칙 보편도입[c -> x]에 의해, 따라서 A의 모든 원소 x에 대해, x는 C의 원소이다.
    """

    # [테스트 3] 복수 주어 분배('모두'), 1항 함수, 등치제거 및 귀류법(부정도입) 증명
    proof_3 = """
    1. 소크라테스와 플라톤은 모두 철학자이라고 가정하자.
    2. 소크라테스는 아테네의 스승과 같다고 가정하자.
    3. 만약 아테네의 스승이 철학자이면, 크산티페의 남편은 현명하다고 가정하자.
    4. 크산티페의 남편은 현명하지 않다고 가정하자.
    5. | 소크라테스는 철학자이라고 가정하자.
    6. | 명제 2, 5로부터 규칙 등치제거[소크라테스 => 아테네의 스승]에 의해, 아테네의 스승은 철학자이다.
    7. | 명제 3, 6으로부터 규칙 조건제거에 의해, 크산티페의 남편은 현명하다.
    8. | 명제 4, 7로부터 규칙 모순도입에 의해, 모순이 발생한다.
    9. 명제 5-8로부터 규칙 부정도입에 의해, 따라서 소크라테스는 철학자가 아니다.
    """

    # [테스트 4] 고유상수 누출(Eigenvalue Leak) 오류 검출 테스트
    proof_error = """
    1. 철학자인 인간 x가 존재한다고 가정하자.
    2. | 새 상수 c에 대해, c는 철학자인 인간이라고 가정하자.
    3. | 명제 2로부터 규칙 연언제거에 의해, c는 철학자이다.
    4. 명제 1, 2-3으로부터 규칙 존재제거에 의해, 따라서 c는 철학자이다.
    """

    for idx, (title, p_text) in enumerate(
        [
            ("증명 1: 동일 주어 생략·후위 존재양화·관형절 분해 종합 증명", proof_1),
            ("증명 2: 관계 명사 영역 제한 양화(집합 포함 추이성) 증명", proof_2),
            ("증명 3: 복수 주어('모두')·함수·등치제거·부정도입 증명", proof_3),
            ("증명 4 [오류 검출 테스트]: 존재제거에서 임시 상수 c 누출 시도", proof_error),
        ],
        start=1,
    ):
        print("=" * 88)
        print(f"▶ {title}")
        print("=" * 88)
        success, logs = verifier.verify_proof(p_text)
        for log in logs:
            print(" ", log)
        print(f"\n  ==> 최종 판정: {'[검증 성공 (VALID PROOF)]' if success else '[검증 실패 (INVALID PROOF)]'}\n")
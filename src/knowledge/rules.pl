% =========================================
% CAPA 1: DERIVADOS BÁSICOS
% =========================================

% Un usuario jugó un juego si tiene interacción
played(U, G) :-
    performedBy(I, U),
    involvesGame(I, G).

% Alto engagement (más de 10 horas)
high_engagement(I) :-
    PlaySession(I),
    duration(I, D),
    D > 10.

% =========================================
% CAPA 2: PREFERENCIAS
% =========================================

% Preferencia por uso
likes_tag(U, T) :-
    performedBy(I, U),
    involvesGame(I, G),
    hasTag(G, T),
    high_engagement(I).

% Preferencia por rating positivo
likes_tag(U, T) :-
    Rating(I),
    ratingValue(I, "Recomendado"),
    performedBy(I, U),
    involvesGame(I, G),
    hasTag(G, T).

% =========================================
% CAPA 3: SIMILARIDAD
% =========================================

% Juegos similares si comparten al menos 2 tags
similar_game(G1, G2) :-
    G1 \= G2,
    hasTag(G1, T1),
    hasTag(G2, T1),
    hasTag(G1, T2),
    hasTag(G2, T2),
    T1 \= T2.

% =========================================
% CAPA 4: SUBGÉNEROS
% =========================================

subgenre(G, t) :-
    hasTag(G, t).

% =========================================
% CAPA 5: CANDIDATOS
% =========================================

% Por preferencias
candidate(U, G) :-
    likes_tag(U, T),
    hasTag(G, T).

% Por similaridad
candidate(U, G2) :-
    played(U, G1),
    similar_game(G1, G2).

% Filtrar juegos ya jugados
valid_candidate(U, G) :-
    candidate(U, G),
    \+ played(U, G).

% =========================================
% CAPA 6: EVENTOS (DESCUENTOS)
% =========================================

discounted_game(G) :-
    hasDiscount(G, D),
    D > 0.3.

candidate(U, G) :-
    discounted_game(G).

% =========================================
% CAPA 7: RECOMENDACIÓN FINAL
% =========================================

recommend(U, G) :-
    valid_candidate(U, G).

% =========================================
% CAPA 8: EXPLICACIONES
% =========================================

% Explicación por tags
explanation(U, G, reason_tag(T)) :-
    recommend(U, G),
    likes_tag(U, T),
    hasTag(G, T).

% Explicación por similaridad
explanation(U, G, reason_similar(G2)) :-
    recommend(U, G),
    played(U, G2),
    similar_game(G2, G).

% Explicación por subgénero
explanation(U, G, reason_subgenre(S)) :-
    recommend(U, G),
    subgenre(G, S).

% Explicación por descuento
explanation(U, G, reason_discount) :-
    recommend(U, G),
    discounted_game(G).

% =========================================
% REGLAS AUXILIARES
% =========================================

% Evitar que un juego sea similar a sí mismo
not_similar(G) :-
    \+ similar_game(G, G).

% ---------- Hechos base: familias de tags RPG ----------
rpg_subgenre(action_rpg,    ['Action RPG','Hack and Slash','Action','Combat']).
rpg_subgenre(jrpg,          ['JRPG','Anime','Turn-Based','Story Rich']).
rpg_subgenre(open_world,    ['Open World','Exploration','Sandbox','Survival']).
rpg_subgenre(tactical_rpg,  ['Tactical RPG','Strategy','Turn-Based Strategy']).
rpg_subgenre(roguelike_rpg, ['Roguelike','Roguelite','Procedural Generation','Dungeon Crawler']).
rpg_subgenre(crpg,          ['RPG','Party-Based RPG','Classic RPG','Isometric']).
rpg_subgenre(soulslike,     ['Souls-like','Difficult','Dark Fantasy','Action']).

% ---------- Penalizaciones / incompatibilidades ----------
incompatible(free_to_play_only, Tag) :-
    \+ Tag = 'Free to Play'.

% ---------- Regla: un juego califica como RPG ----------
is_rpg(Tags) :-
    rpg_subgenre(_, SubTags),
    member(T, SubTags),
    member(T, Tags), !.

% ---------- Regla: subgénero dominante del juego ----------
dominant_subgenre(Tags, Subgenre) :-
    rpg_subgenre(Subgenre, SubTags),
    include(member_of(SubTags), Tags, Matches),
    length(Matches, N),
    N > 0, !.

member_of(List, Elem) :- member(Elem, List).

% ---------- Boost por alta coincidencia de tags ----------
tag_overlap_score(GameTags, PreferredTags, Score) :-
    include(member_of(PreferredTags), GameTags, Common),
    length(Common, C),
    length(PreferredTags, P),
    (P > 0 -> Score is C / P ; Score is 0).

% ---------- Penalización por tags no deseados ----------
dislike_penalty(GameTags, DislikedTags, Penalty) :-
    include(member_of(DislikedTags), GameTags, Bad),
    length(Bad, B),
    Penalty is B * 0.15.

% ---------- Regla: juego recomendable ----------
% Un juego es recomendable si:
%   1. Es un RPG
%   2. No contiene demasiados tags no deseados
%   3. Cumple el filtro de precio (si aplica)
recommendable(Tags, DislikedTags, MaxPrice, Price) :-
    is_rpg(Tags),
    dislike_penalty(Tags, DislikedTags, Penalty),
    Penalty < 0.45,
    (MaxPrice > 0 -> Price =< MaxPrice ; true).

% ---------- Clasificación de precio ----------
price_tier(Price, free)       :- Price =:= 0.
price_tier(Price, budget)     :- Price > 0,  Price =< 10.
price_tier(Price, mid_range)  :- Price > 10, Price =< 30.
price_tier(Price, premium)    :- Price > 30, Price =< 60.
price_tier(Price, deluxe)     :- Price > 60.

% ---------- Clasificación de rating ----------
rating_tier(R, masterpiece) :- R >= 9.0.
rating_tier(R, excellent)   :- R >= 8.0, R < 9.0.
rating_tier(R, good)        :- R >= 7.0, R < 8.0.
rating_tier(R, mixed)       :- R >= 5.0, R < 7.0.
rating_tier(R, poor)        :- R < 5.0.
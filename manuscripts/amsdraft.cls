%% amsdraft.cls -- PREVIEW ONLY. NOT FOR SUBMISSION.
%%
%% The real AMS class, ametsocV6.cls, has to be downloaded by hand from
%% https://www.ametsoc.org/ams/publications/author-information/latex-author-info/
%% ("AMS LaTeX files"). The AMS web server returns 403 to anything that is not
%% an interactive browser, so it cannot be fetched by script.
%%
%% This class exists so the manuscript can be read and edited before that file
%% arrives, whether that is on a laptop or in Overleaf. It defines the handful
%% of AMS commands catalogue_paper.tex uses and nothing else, on top of plain
%% article. Margins, spacing, the title block and the reference format are all
%% WRONG relative to a real AMS submission.
%%
%% No edit is needed to stop using it. catalogue_paper.tex picks a real AMS
%% class as soon as one is present, and this file goes inert. Leave it out of
%% the submission bundle.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{amsdraft}[2026/09/24 Preview stand-in for ametsocV6]
\LoadClass[12pt]{article}

\RequirePackage[margin=1in]{geometry}
\RequirePackage{setspace}
\RequirePackage{natbib}
\onehalfspacing

% --- Title block -----------------------------------------------------------
\newcommand{\correspondingauthor}[1]{\thanks{Corresponding author: #1}}
\newcommand{\authors}[1]{\author{#1}}
% The AMS class pairs these to add authors at a second affiliation. Stubbing
% them out would silently drop a co-author from the preview, so they append to
% the author block instead.
% No \makeatletter here: inside a .cls, @ is already a letter, and adding a
% \makeatother would switch it off for the rest of this file.
\newcommand{\extraauthor}[1]{\g@addto@macro\@author{\\[2pt] #1}}
\newcommand{\extraaffil}[1]{\g@addto@macro\@author{\\ \textit{#1}}}
\newcommand{\affiliation}[1]{\date{#1}}

% AMS takes the abstract as a command argument, article as an environment.
\renewcommand{\abstract}[1]{%
  \gdef\amsdraft@abstract{#1}}
\let\amsdraft@maketitle\maketitle
\renewcommand{\maketitle}{%
  \amsdraft@maketitle
  \@ifundefined{amsdraft@abstract}{}{%
    \begin{center}\textbf{Abstract}\end{center}%
    \begin{quotation}\noindent\amsdraft@abstract\end{quotation}\bigskip}}

% --- Sectioning the AMS class adds ----------------------------------------
\newcommand{\statement}{\section*{Significance statement}}
\newcommand{\capsule}{\section*{Capsule summary}}
\newcommand{\datastatement}{\section*{Data availability statement}}
\newcommand{\appendixtitle}[1]{\section*{Appendix: #1}}
\newcommand{\appendcaption}[2]{\caption{#2}}

% --- Table rules the AMS class provides -----------------------------------
\RequirePackage{booktabs}
\newcommand{\topline}{\toprule}
\newcommand{\midline}{\midrule}
\newcommand{\botline}{\bottomrule}

\endinput

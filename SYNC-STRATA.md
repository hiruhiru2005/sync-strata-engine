# Sync-Strata engine

[English](#english) | [日本語](#日本語)

## English

This repository is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors.
It holds the engine of [Sync-Strata](https://github.com/hiruhiru2005/sync-strata), with upstream's history and authors kept.
For Strata itself, go to [Niko1221/Strata](https://github.com/Niko1221/Strata).

Thank you to Niko1221 and everyone who built Strata. Sync-Strata would not exist without it.

### Branches and tags

| Name | What it is |
|---|---|
| `sync/windows` | The engine source of the latest Windows release |
| `sync/linux` | The engine source of the latest Linux release |
| `sync-<version>-windows`, `sync-<version>-linux` | The engine source of each release (`sync-0.1.0` for both) |

Both branches start from upstream Strata 0.2.7 (`5e8a4cca`). The commits after it are Sync-Strata's changes.
Ten of them are upstream fixes taken in later; their author is kept.

### Commit IDs in earlier notes

Release notes and the sync-strata repository quote commit IDs from the author's working repository (for example `762a3b05`).
Before this fork was published, the author name and e-mail address in Sync-Strata's commits were changed to this account.
The contents of the commits are the same, but their IDs changed:

| Release | Windows (notes) | Windows (here) | Linux (notes) | Linux (here) |
|---|---|---|---|---|
| 0.1.0 | `8376e999` | `fb85ead4a3` (`sync-0.1.0`) | same as Windows | same as Windows |
| 0.1.1 | `9fa44244` | `3bc2de762e` | `2d6076ad` | `3e229a855d` |
| 0.1.2 | `d278210e` | `508caf2365` | `2c9fe533` | `4ff1a157ab` |
| 0.1.3, 0.1.4 | `762a3b05` | `10f0d35fd4` | `829f5c13` | `f31f2f8be5` |

### Earlier copy in sync-strata

Until November 2026, the sync-strata repository carried this engine as one squashed copy under `engine/`, without upstream's history and authors.
This fork keeps them, and `engine/` in sync-strata now points here as a submodule.

### License

MIT, the same as upstream (`LICENSE`). Sync-Strata's changes: Copyright (c) 2026 hiruhiru2005.

## 日本語

このリポジトリは、Niko1221 さんと Strata の貢献者の皆さんによる [Strata](https://github.com/Niko1221/Strata) のフォークです。
[Sync-Strata](https://github.com/hiruhiru2005/sync-strata) のエンジンを、本家の履歴と作者を残したまま置いています。
Strata そのものは [Niko1221/Strata](https://github.com/Niko1221/Strata) をご覧ください。

Strata を作った Niko1221 さんと貢献者の皆さんに感謝します。Strata がなければ Sync-Strata はありませんでした。

### 枝とタグ

| 名前 | 中身 |
|---|---|
| `sync/windows` | 最新の Windows 版のエンジンのソース |
| `sync/linux` | 最新の Linux 版のエンジンのソース |
| `sync-<版>-windows`、`sync-<版>-linux` | 各版のエンジンのソース（0.1.0 は両方とも `sync-0.1.0`） |

どちらの枝も本家 Strata 0.2.7（`5e8a4cca`）から始まり、その後のコミットが Sync-Strata の変更です。
そのうち 10 個は、後から取り込んだ本家の直しで、作者はそのまま残しています。

### これまでの説明に出てくるコミットの番号

リリースの説明や sync-strata のリポジトリには、作者の作業用のリポジトリのコミットの番号（例: `762a3b05`）が書かれています。
このフォークを公開する前に、Sync-Strata のコミットの作者の名前とメールをこのアカウントに書き換えたので、中身は同じですが番号が変わりました。対応は上の表のとおりです。

### sync-strata にあった写し

2026 年 11 月まで、sync-strata のリポジトリは、このエンジンを本家の履歴と作者を残さずに `engine/` へ 1 つのコミットとしてまとめて入れていました。
このフォークで履歴と作者を残し、sync-strata の `engine/` はこのフォークを指すサブモジュールに替えました。

### ライセンス

本家と同じ MIT です（`LICENSE`）。Sync-Strata の変更分は Copyright (c) 2026 hiruhiru2005 です。

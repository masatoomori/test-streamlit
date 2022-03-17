# test-streamlit

## 準備

### パッケージインストール

```bash
pip install streamlit
pip install watchdog
```

### 設定ファイルの作成

```bash
$ mkdir -p ~/.streamlit
$ cat << 'EOF' > ~/.streamlit/credentials.toml
[general]
email = ""
EOF
```

## 実行

```bash
streamlit run app.py
```

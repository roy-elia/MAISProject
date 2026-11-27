"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { useCallback, useEffect, useRef, useState } from "react"
import { Calendar, ExternalLink, RotateCw, Sparkles } from "lucide-react"
import { getGitHubRepoUrl } from "@/lib/config"

// The regression model counts years from 2006 and covers 2006-2024.
const FIRST_YEAR = 2006
const LAST_YEAR = 2024
// Mean absolute error on the validation split (trainer_state.json, checkpoint-684).
const VALIDATION_MAE_YEARS = 4.3
// Stop polling after this long and offer a manual retry.
const WAKE_LIMIT_MS = 4 * 60 * 1000
const POLL_INTERVAL_MS = 3000

const EXAMPLES = [
  "Just upgraded to Windows Vista, loving the new Aero look",
  "Ron Paul 2008! The digg front page is all about it",
  "Anyone else stuck at home in quarantine? Zoom calls all day",
  "This meme is fire fr fr no cap",
]

type ServerStatus = "checking" | "waking" | "ready" | "unavailable"

type Prediction = {
  predicted_year: number
  year_index: number
}

export function Demo() {
  const [inputText, setInputText] = useState("")
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [queuedText, setQueuedText] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [serverStatus, setServerStatus] = useState<ServerStatus>("checking")
  const [wakeSeconds, setWakeSeconds] = useState(0)
  const pollRun = useRef(0)

  // Poll the backend until it reports the model is loaded. The free host sleeps
  // when idle, so the first visit of the day can take a minute or more.
  const waitForServer = useCallback(async () => {
    const run = ++pollRun.current
    const started = Date.now()
    setServerStatus((s) => (s === "ready" ? s : "checking"))
    while (run === pollRun.current) {
      try {
        const res = await fetch("/api/health", { cache: "no-store" })
        if (res.ok) {
          if (run === pollRun.current) setServerStatus("ready")
          return
        }
      } catch {
        // Network hiccup: keep polling.
      }
      if (run !== pollRun.current) return
      const elapsed = Date.now() - started
      if (elapsed > WAKE_LIMIT_MS) {
        setServerStatus("unavailable")
        return
      }
      setServerStatus("waking")
      setWakeSeconds(Math.round(elapsed / 1000))
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
    }
  }, [])

  useEffect(() => {
    waitForServer()
    return () => {
      pollRun.current++
    }
  }, [waitForServer])

  const runPrediction = useCallback(
    async (text: string) => {
      setIsPredicting(true)
      setErrorMessage(null)
      try {
        for (let attempt = 0; attempt < 3; attempt++) {
          const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, task_type: "regression" }),
          })
          const data = await res.json().catch(() => ({}))
          if (res.ok && typeof data.predicted_year === "number") {
            setPrediction({ predicted_year: data.predicted_year, year_index: data.year_index })
            return
          }
          if (res.status !== 503) {
            setErrorMessage(typeof data.error === "string" ? data.error : "The prediction failed. Please try again.")
            return
          }
          // The server went back to sleep or is restarting: wait for it, then retry.
          setServerStatus("waking")
          await waitForServer()
        }
        setErrorMessage("The model server is still starting. Please try again in a minute.")
      } catch {
        setErrorMessage("Could not reach the server. Check your connection and try again.")
      } finally {
        setIsPredicting(false)
      }
    },
    [waitForServer],
  )

  // A click while the server is waking is queued and sent once it is ready.
  useEffect(() => {
    if (queuedText && serverStatus === "ready" && !isPredicting) {
      const text = queuedText
      setQueuedText(null)
      runPrediction(text)
    }
  }, [queuedText, serverStatus, isPredicting, runPrediction])

  const handlePredict = (text = inputText) => {
    const trimmed = text.trim()
    if (!trimmed || isPredicting) return
    setPrediction(null)
    setErrorMessage(null)
    if (serverStatus === "ready") {
      runPrediction(trimmed)
    } else {
      setQueuedText(trimmed)
      if (serverStatus === "unavailable") waitForServer()
    }
  }

  const pickExample = (text: string) => {
    setInputText(text)
    handlePredict(text)
  }

  const busy = isPredicting || queuedText !== null
  const estimate = prediction ? FIRST_YEAR + prediction.year_index : null
  const markerPercent =
    estimate === null
      ? 0
      : Math.min(100, Math.max(0, ((estimate - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR)) * 100))

  return (
    <section id="demo" className="py-16 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-accent/5" />

      <div className="max-w-5xl mx-auto relative z-10">
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="p-3 rounded-full bg-accent/10 border border-accent/20">
            <Sparkles className="h-8 w-8 text-accent" />
          </div>
          <h2 className="text-3xl font-bold text-center bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Interactive Demo
          </h2>
        </div>

        <Card className="p-6 sm:p-8 border-2 relative overflow-hidden">
          <div className="space-y-6">
            <ServerStatusBadge status={serverStatus} wakeSeconds={wakeSeconds} onRetry={waitForServer} />

            <div>
              <label htmlFor="comment-input" className="block text-sm font-medium mb-2">
                Enter a Reddit-style comment:
              </label>
              <Textarea
                id="comment-input"
                placeholder="Type a comment, or pick an example below"
                value={inputText}
                onChange={(e) => {
                  setInputText(e.target.value)
                  if (prediction) setPrediction(null)
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handlePredict()
                }}
                rows={4}
                maxLength={2000}
                className="resize-none"
              />
              <div className="mt-3 flex flex-wrap gap-2" aria-label="Example comments">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => pickExample(example)}
                    disabled={busy}
                    className="text-xs px-3 py-1.5 rounded-full border border-border bg-secondary/50 text-muted-foreground hover:text-foreground hover:border-accent/50 transition-colors disabled:opacity-50 text-left"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            <Button onClick={() => handlePredict()} disabled={!inputText.trim() || busy} size="lg" className="w-full gap-2">
              <Calendar className="h-5 w-5" />
              {queuedText !== null
                ? "Waiting for the model server…"
                : isPredicting
                  ? "Predicting…"
                  : "Predict Year"}
            </Button>

            {errorMessage && (
              <p className="text-sm text-destructive text-center" role="alert">
                {errorMessage}
              </p>
            )}

            {prediction && estimate !== null && (
              <div
                className="p-6 sm:p-8 bg-gradient-to-br from-accent/15 via-accent/5 to-transparent border-2 border-accent/30 rounded-2xl"
                aria-live="polite"
              >
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center">
                  Predicted year
                </p>
                <p className="text-6xl font-bold text-accent text-center mt-2">{prediction.predicted_year}</p>
                <p className="text-sm text-muted-foreground text-center mt-2">
                  Model estimate {(Math.round(estimate * 10 + 1e-6) / 10).toFixed(1)} · typical error ±{VALIDATION_MAE_YEARS} years
                </p>

                <div className="mt-6" aria-hidden="true">
                  <div className="relative h-2 rounded-full bg-muted">
                    <div
                      className="absolute top-0 h-2 rounded-full bg-accent/25"
                      style={{
                        left: `${Math.max(0, markerPercent - (VALIDATION_MAE_YEARS / (LAST_YEAR - FIRST_YEAR)) * 100)}%`,
                        right: `${Math.max(0, 100 - markerPercent - (VALIDATION_MAE_YEARS / (LAST_YEAR - FIRST_YEAR)) * 100)}%`,
                      }}
                    />
                    <div
                      className="absolute -top-1 h-4 w-4 -ml-2 rounded-full bg-accent border-2 border-background shadow"
                      style={{ left: `${markerPercent}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground mt-2">
                    <span>{FIRST_YEAR}</span>
                    <span>2015</span>
                    <span>{LAST_YEAR}</span>
                  </div>
                </div>
              </div>
            )}

            <div className="pt-4 border-t border-border">
              <p className="text-sm text-muted-foreground mb-4">
                <strong className="text-foreground">Note:</strong> This demo runs the regression model, which estimates
                a single year between <strong className="text-foreground">{FIRST_YEAR}</strong> and{" "}
                <strong className="text-foreground">{LAST_YEAR}</strong>. It was trained on a small sample, so its
                estimates cluster in the mid-2010s. The server runs on free hosting and sleeps when idle; the first
                prediction after a quiet spell can take a minute or two.
              </p>
              <Button variant="outline" size="sm" className="gap-2 bg-transparent" asChild>
                <a href={getGitHubRepoUrl()} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4" />
                  View on GitHub
                </a>
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </section>
  )
}

function ServerStatusBadge({
  status,
  wakeSeconds,
  onRetry,
}: {
  status: ServerStatus
  wakeSeconds: number
  onRetry: () => void
}) {
  if (status === "ready") {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
        Model server ready
      </div>
    )
  }
  if (status === "unavailable") {
    return (
      <div className="flex flex-wrap items-center gap-2 text-sm text-destructive" role="status">
        <span className="h-2.5 w-2.5 rounded-full bg-destructive" />
        The model server did not respond.
        <button type="button" onClick={onRetry} className="inline-flex items-center gap-1 underline">
          <RotateCw className="h-3.5 w-3.5" /> Try again
        </button>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
      <span className="h-2.5 w-2.5 rounded-full bg-amber-500 animate-pulse" />
      {status === "checking"
        ? "Connecting to the model server…"
        : `Waking the model server (${wakeSeconds} s). It sleeps when idle; this takes a minute or two. You can type and press Predict now.`}
    </div>
  )
}

import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"

// The backend runs on a free host that sleeps when idle and takes a minute or
// more to wake. Each check waits only a few seconds, so the page can poll this
// route and show progress instead of hanging on one long request.
export async function GET() {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  try {
    const response = await fetch(`${backendUrl}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(8000),
    })
    const data = await response.json().catch(() => ({}))
    const ready = response.ok && data?.model_loaded === true
    return NextResponse.json({ status: ready ? "ready" : "waking" }, { status: ready ? 200 : 503 })
  } catch {
    return NextResponse.json({ status: "waking" }, { status: 503 })
  }
}

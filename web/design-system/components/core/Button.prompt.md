Button — the CiteClaw action control; `primary` is the green theme button (reserve for the one key action per view), `secondary` is a cream outlined button, `ghost` is borderless.

```jsx
<Button variant="primary" icon={<PlayIcon/>}>Run pipeline</Button>
<Button variant="secondary">Pause</Button>
<Button variant="ghost" size="sm">Clear draft</Button>
```

Sizes: sm / md / lg. Pass `disabled`. Green is critical-only — don't use primary for routine controls.
